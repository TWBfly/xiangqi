# Pikafish 非法局面防护设计

日期：2026-07-22
目标平台：Windows 10、macOS 本地回归、Pikafish UCI 引擎

## 目标

彻底阻止棋盘识别产生的非法兵卒坐标进入 Pikafish，避免官方引擎因 `position fen` 不合法而主动终止。修复必须覆盖识别源头、棋盘总校验、FEN 序列化和引擎进程边界，不降低现有识别置信度或局面合法性标准。

## 现场证据

故障 FEN 为：

```text
2bakab1r/9/7c1/pp4p1p/2p6/9/P1P1P1P1P/2N1C4/9/2BAKABNR w - - 0 1
```

其中黑卒位于内部坐标 `(3, 1)`。黑卒在第 3、4 行尚未过河，只能处于第 0、2、4、6、8 列，因此该坐标不可能由合法对局产生。

当前本地代码对该 FEN 的行为是：

```text
is_coordinate_legal("p", 3, 1) == True
validate_board(board) == ""
board_to_fen(board, "w") == 原始非法 FEN
```

同一 FEN 送入仓库内 `pikafish_bin` 后稳定以退出码 1 结束，并输出：

```text
Unsupported position. BLACK pawn(s) on invalid positions.
```

这排除了 Windows 进程启动、CPU 指令集、NNUE 路径和搜索超时。故障是本地合法性模型与 Pikafish 官方模型不一致。

## 官方规则

Pikafish 当前官方主线提交 `97133eebb6ed55e0bfa13262555c77d683d6ac0f` 在 `src/bitboard.h` 中定义：

- 未过河黑卒只允许第 3、4 行的 A/C/E/G/I 五路，过河后允许所有列；
- 未过河红兵只允许第 5、6 行的 A/C/E/G/I 五路，过河后允许所有列。

`src/position.cpp` 的 `Position::set()` 对兵、士、象、将帅应用 `ValidBB`，发现越界棋子就返回 `PositionSetError`。`src/uci.cpp` 将 `position` 命令错误升级为 critical error 并终止进程。

本地 Pikafish 提交 `7c09ec2578c3b7a48ce4ee5cd067971ef9c4c320` 虽比主线落后 5 个提交，但上述规则及退出行为一致。升级引擎不能解决应用发送非法 FEN 的问题，本次不修改嵌套 `pikafish` 仓库。

## 根因

故障由四道防线同时缺失构成：

1. `is_coordinate_legal()` 只限制兵卒不能退回起始线后方，没有限制未过河时的列。
2. `validate_board()` 重复实现士、象坐标规则，却完全遗漏兵卒坐标。
3. `board_to_fen()` 只检查矩阵尺寸和棋子字符，不检查局面合法性。
4. `PikafishEngine.get_best_move()` 不预检 FEN；`AssistantApp.best_move()` 还会对确定性非法输入重启引擎并重试。

另有一个诊断竞态：输出线程收到 EOF 时，`process.poll()` 可能暂时仍为 `None`，所以真实退出码 1 被显示为“未知”。它不造成进程退出，但妨碍定位后续真实故障。

## 采用方案：共享规则与两层边界防御

### 1. 单一坐标规则

继续复用现有 `is_coordinate_legal(label, row, col)`，不创建新规则模块。补全兵卒规则：

```python
black_pawn_legal = row >= 3 and (row >= 5 or col % 2 == 0)
red_pawn_legal = row <= 6 and (row <= 4 or col % 2 == 0)
```

识别器现有两个候选评分路径已经调用该函数，因此非法位置的兵卒模板分数会变成 0，分类器会选择下一个合法棋子类型或空位。

### 2. 总棋盘校验复用共享规则

`validate_board()` 在保留棋子数量、九宫、将帅照面校验后，遍历所有非空格并调用 `is_coordinate_legal()`。删除函数内部重复的士、象坐标集合，让识别分类和总校验使用同一规则来源。

非法坐标错误必须包含棋子名称和零基坐标，例如：

```text
黑卒在非法坐标: (3, 1)
```

### 3. FEN 序列化边界

`board_to_fen()` 在编码前调用 `validate_board()`，有错误就抛出 `ValueError`。这样 `AssistantApp.sync_current_board()` 会在创建分析线程之前通过现有异常路径提示“无法同步并分析”，不会启动或重启 Pikafish。

### 4. 引擎进程边界

`PikafishEngine.get_best_move()` 同时兼容纯 FEN、`position fen <fen>` 和 `position fen <fen> moves ...`。它从命令中提取六字段基础 FEN，调用现有 `fen_to_board()` 预检，再发送 UCI 命令。

预检失败抛出带上下文的 `ValueError`，且不发送 `ucinewgame`、`isready`、`position` 或 `go`。`AssistantApp.best_move()` 对该确定性输入错误直接上抛，不执行两次引擎重启；对进程异常、超时等瞬时故障继续保留当前重试策略。

这一层并不复制象棋规则，只复用 `fen_to_board()`，用于保护直接调用引擎包装器、绕过 `board_to_fen()` 的路径。

### 5. 退出码诊断

当 `_read_until()` 从输出队列读到 EOF 时，短暂调用现有子进程的 `wait()` 获取最终退出码；若仍无法取得，再显示“未知”。这只修复诊断信息，不改变引擎生命周期或重试次数。

## 数据流

```text
截图
→ Calibration.recognize
→ is_coordinate_legal：非法兵卒候选分数归零
→ validate_board：共享坐标规则总校验
→ board_to_fen：序列化前再次总校验
→ PikafishEngine.get_best_move：提取并解析基础 FEN
→ UCI position / go
```

任何一层拒绝后，数据流立即停止。只有通过同一合法性规则的棋盘才能到达 Pikafish。

## 文件范围

- 修改 `board_recognition.py`：补全兵卒坐标规则、统一总校验、FEN 序列化前校验。
- 修改 `pikafish_engine.py`：发送前解析基础 FEN、拒绝非法输入、稳定读取退出码。
- 修改 `app.py`：确定性 `ValueError` 不重试；更新可见版本号。
- 修改 `test_assistant.py`：增加规则、故障 FEN、进程存活、带 moves 命令和退出码回归。
- 不修改 `game_state.py`、`capture.py`、依赖列表或嵌套 `pikafish` 仓库。

当前四个目标文件已有用户未提交修改。实施只能应用精确增量补丁，不覆盖、不格式化和不回退无关内容。

## TDD 方案

依次建立并观察以下失败：

1. 黑卒 `(3, 1)`、`(4, 1)` 非法，过河后的 `(5, 1)` 合法；红兵规则镜像对称。
2. `validate_board()` 和 `fen_to_board()` 拒绝现场故障 FEN，并包含 `(3, 1)`。
3. `board_to_fen()` 拒绝含非法兵卒的棋盘。
4. `get_best_move()` 对非法纯 FEN 和带 `moves` 命令均在发送前失败，现有引擎进程保持存活。
5. 合法标准开局和现有合法中局仍能获得 Pikafish 合法着法。
6. EOF 到达但首次 `poll()` 未更新时，错误信息仍能显示最终退出码。

每组测试先在未修改生产代码时出现与缺陷一致的 RED，再写最小实现进入 GREEN。

## 验收标准

- 现场非法 FEN 在本地被拒绝，错误明确指出黑卒 `(3, 1)` 非法。
- 非法局面不会发送给 Pikafish，拒绝后已经启动的 Pikafish 进程仍存活。
- 黑卒和红兵过河前后、偶数列和奇数列边界与官方 `PawnBB` 一致。
- `position fen ... moves ...` 合法命令继续工作。
- 合法标准开局和已有合法中局实机分析继续返回合法 UCI 着法。
- 瞬时引擎故障仍可重试，确定性输入错误不重试。
- 不降低识别阈值，不增加依赖，不修改官方引擎。
- 完整单元测试、Python 语法编译、现场 FEN 回归和真实 Pikafish 回归通过。

## 非目标

- 不尝试让 Pikafish接受非法局面。
- 不自动修改或“猜测修正”一份已经成型的非法 FEN；识别阶段只排除不可能的棋子标签，边界阶段只拒绝。
- 不重构整个棋盘领域模型或拆分新模块。
- 不借本次修复调整模板阈值、圆检测、棋盘对齐或搜索强度。
