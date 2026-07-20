# 象棋任意局面、完整历史与失步恢复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 让辅助器从标准开局或任意合法中盘启动，完整记录实际走法，正确处理重复局面、悔棋和失步重同步，并用可配置搜索时间分析真实历史局面。

**Architecture:** 新增 `GameState` 作为唯一棋局语义状态源；视觉层只产生候选棋盘，应用层将候选提交给状态对象，引擎消费 `position fen ... moves ...`。未知历史无法反推时，以识别局面或用户 FEN 建立新的历史基点。

**Tech Stack:** Python 3、Tkinter、OpenCV、NumPy、Pikafish UCI、unittest；不增加依赖。

## Global Constraints

- 同一棋盘皮肤必须至少完成过一次标准开局校准。
- 默认搜索 5 秒，可配置范围为 1～60 秒。
- 输出是搜索预算内最佳着法，不承诺数学意义上的绝对最优。
- 不伪造从最终盘面无法唯一确定的丢失历史。
- 视觉结果不确定时不得改变逻辑状态。
- 项目目录不是 Git 仓库，所有提交步骤替换为文件范围复查。

---

### Task 1: FEN 解析、棋盘校验和 UCI 坐标

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Produces: `fen_to_board(fen: str) -> tuple[tuple, str]`
- Produces: `move_to_uci(move: Move) -> str`
- Produces: `validate_board(board: tuple) -> str`

- [x] **Step 1: 写失败测试**

```python
def test_fen_round_trip_preserves_board_and_active_side(self):
    fen = board_to_fen(STANDARD_BOARD, "b")
    board, side = fen_to_board(fen)
    self.assertEqual(board, STANDARD_BOARD)
    self.assertEqual(side, "b")

def test_fen_rejects_facing_kings(self):
    with self.assertRaisesRegex(ValueError, "将帅不能照面"):
        fen_to_board("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")

def test_move_to_uci_uses_pikafish_coordinates(self):
    self.assertEqual(move_to_uci(Move("w", (9, 7), (7, 6))), "h0g2")
```

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.BoardTests`

Expected: FAIL，三个新接口尚不存在。

- [x] **Step 3: 实现最小解析与校验**

在 `board_recognition.py` 中复用 `normalize_board`，实现：

```python
def move_to_uci(move):
    sr, sc = move.start
    er, ec = move.end
    return f"{chr(97 + sc)}{9 - sr}{chr(97 + ec)}{9 - er}"


def validate_board(board):
    board = normalize_board(board)
    pieces = [piece for row in board for piece in row if piece]
    if pieces.count("K") != 1 or pieces.count("k") != 1:
        return "红帅或黑将数量错误"
    limits = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}
    if any(pieces.count(piece) > limit or pieces.count(piece.lower()) > limit
           for piece, limit in limits.items()):
        return "棋子数量超过上限"
    red_king = next((pos for pos in ((r, c) for r in range(10) for c in range(9))
                     if board[pos[0]][pos[1]] == "K"), None)
    black_king = next((pos for pos in ((r, c) for r in range(10) for c in range(9))
                       if board[pos[0]][pos[1]] == "k"), None)
    if red_king[0] not in range(7, 10) or red_king[1] not in range(3, 6):
        return "红帅不在九宫内"
    if black_king[0] not in range(0, 3) or black_king[1] not in range(3, 6):
        return "黑将不在九宫内"
    if red_king[1] == black_king[1] and not any(
        board[row][red_king[1]] for row in range(black_king[0] + 1, red_king[0])
    ):
        return "将帅不能照面"
    return ""
```

`fen_to_board` 必须验证 6 个字段、10×9 盘面、字符集合、`w/b`、`- -`、非负半回合数和正全回合数，并调用 `validate_board`。

- [x] **Step 4: 复用校验并运行测试**

让 `Calibration._validate` 复用 `validate_board` 后运行：

Run: `python3 -m unittest -v test_assistant.BoardTests test_assistant.RecognitionTests`

Expected: PASS；真实截图附件允许 1 个 SKIP。

### Task 2: 单一棋局状态、完整历史和版本

**Files:**
- Create: `game_state.py`
- Test: `test_assistant.py`

**Interfaces:**
- Produces: `Snapshot(board, side_to_move, moves_length)`
- Produces: `GameState.start(board, side_to_move, player, base_fen=None)`
- Produces: `apply_board(board) -> bool`
- Produces: `restore_snapshot(board) -> bool`
- Produces: `resync(board, side_to_move, base_fen=None) -> None`
- Produces: `needs_analysis() -> bool`
- Produces: `mark_analyzed() -> None`
- Produces: `position_command() -> str`

- [x] **Step 1: 写状态失败测试**

覆盖以下断言：

```python
state = GameState.start(STANDARD_BOARD, "w", "w")
after_red = moved(STANDARD_BOARD, (9, 7), (7, 6))
self.assertTrue(state.apply_board(after_red))
self.assertEqual(state.moves, ["h0g2"])
self.assertEqual(state.side_to_move, "b")
self.assertEqual(state.position_version, 1)

after_black = moved(after_red, (0, 1), (2, 2))
state.apply_board(after_black)
version = state.position_version
self.assertTrue(state.restore_snapshot(STANDARD_BOARD))
self.assertEqual(state.moves, [])
self.assertEqual(state.side_to_move, "w")
self.assertGreater(state.position_version, version)
```

另写重复局面测试：恢复到旧棋盘后 `needs_analysis()` 必须基于新版本返回真，不能按棋盘永久去重。

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.GameStateTests`

Expected: ERROR，`game_state` 尚不存在。

- [x] **Step 3: 实现 `GameState`**

```python
from dataclasses import dataclass

from board_recognition import board_to_fen, detect_move, move_to_uci, normalize_board


@dataclass(frozen=True)
class Snapshot:
    board: tuple
    side_to_move: str
    moves_length: int


class GameState:
    @classmethod
    def start(cls, board, side_to_move, player, base_fen=None):
        return cls(board, side_to_move, player, base_fen)

    def __init__(self, board, side_to_move, player, base_fen=None):
        if side_to_move not in {"w", "b"} or player not in {"w", "b"}:
            raise ValueError("行棋方必须是w或b")
        self.board = normalize_board(board)
        self.side_to_move = side_to_move
        self.player = player
        self.base_fen = base_fen or board_to_fen(self.board, side_to_move)
        self.moves = []
        self.snapshots = [Snapshot(self.board, side_to_move, 0)]
        self.position_version = 0
        self.last_analyzed_version = None
```

`apply_board` 必须调用现有 `detect_move`，只接受 `move.side == side_to_move`；成功后追加 UCI、切换行棋方、保存快照并递增版本。`restore_snapshot` 从后向前匹配、截断 moves 和 snapshots、恢复行棋方并递增版本。`resync` 清空旧历史并建立新基点。

- [x] **Step 4: 实现分析去重和命令生成**

```python
def needs_analysis(self):
    return (
        self.side_to_move == self.player
        and self.last_analyzed_version != self.position_version
    )

def mark_analyzed(self):
    self.last_analyzed_version = self.position_version

def position_command(self):
    suffix = f" moves {' '.join(self.moves)}" if self.moves else ""
    return f"position fen {self.base_fen}{suffix}"
```

- [x] **Step 5: 运行状态测试**

Run: `python3 -m unittest -v test_assistant.GameStateTests`

Expected: PASS。

### Task 3: Pikafish 消费完整历史和可配置预算

**Files:**
- Modify: `pikafish_engine.py`
- Modify: `app.py`
- Test: `test_assistant.py`

**Interfaces:**
- Change: `PikafishEngine.get_best_move(position_command: str, movetime=5000)`
- Change: `AssistantApp.best_move(position_command: str, movetime: int)`

- [x] **Step 1: 写失败测试**

使用真实的 `_lines` 队列和替代的 `send_command`，断言：

```python
command = "position fen " + board_to_fen(STANDARD_BOARD, "w") + " moves h0g2"
engine.get_best_move(command, movetime=5000)
self.assertEqual(sent[:2], [command, "go movetime 5000"])
```

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.EngineTests.test_engine_uses_complete_position_history`

Expected: FAIL，旧代码会再次添加 `position fen`。

- [x] **Step 3: 修改引擎入口**

```python
def get_best_move(self, position_command, movetime=5000):
    if not position_command.startswith("position fen "):
        raise ValueError("无效Pikafish局面命令")
    with self._lock:
        self.send_command(position_command)
        self.send_command(f"go movetime {int(movetime)}")
        lines = self._read_until(
            lambda line: line.startswith("bestmove "),
            max(5.0, movetime / 1000 + 3.0),
        )
    # 保留现有 bestmove 和 score 解析
```

更新所有调用者和真实引擎测试，统一传完整 position 命令。

- [x] **Step 4: 运行引擎测试**

Run: `python3 -m unittest -v test_assistant.EngineTests`

Expected: PASS。

### Task 4: 视觉失步信号和重置接口

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Extend: `Recognition(..., recovery: bool = False)`
- Produces: `MotionTracker.reset(frame, board) -> None`
- Behavior: 连续 3 个无法解释的稳定变化后尝试全盘识别并返回 `recovery=True`。

- [x] **Step 1: 写失败测试**

```python
tracker = MotionTracker(calibration)
tracker.observe(initial, "w")
for _ in range(2):
    self.assertFalse(tracker.observe(unknown, "w").recovery)
self.assertTrue(tracker.observe(unknown, "w").recovery)
tracker.reset(unknown, recognized_board)
self.assertEqual(tracker.board, recognized_board)
```

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.RecognitionTests.test_motion_requests_recovery_after_three_unresolved_frames`

Expected: FAIL，`Recognition` 没有 recovery 字段。

- [x] **Step 3: 实现计数和重置**

为 `MotionTracker` 增加 `unresolved_count`。只有 `len(changed) >= 2`、不是高亮刷新且无法推导着法时才递增；其他路径清零。第三次调用 `calibration.recognize(frame)`，有效时返回其棋盘和 `recovery=True`，但不自行提交逻辑状态。

`reset(frame, board)` 必须同时设置 `board/reference`，并清空 candidate、pending、last_move_cells、refresh 和 unresolved 状态。

- [x] **Step 4: 运行识别测试**

Run: `python3 -m unittest -v test_assistant.RecognitionTests`

Expected: PASS；真实截图附件允许 1 个 SKIP。

### Task 5: 应用层接入 GameState、悔棋和失步恢复

**Files:**
- Modify: `app.py`
- Test: `test_assistant.py`

**Interfaces:**
- `run_loop(player, state, initial_frame)` 使用 `GameState`，不再创建 `TurnTracker`。
- 正常候选调用 `state.apply_board`。
- `recovery=True` 时先调用 `state.restore_snapshot`；未知棋盘停止跟踪并发送 `("needs_sync", board)`。

- [x] **Step 1: 写失败集成测试**

模拟以下事件序列并断言：

```text
初始版本分析一次
用户偏离推荐走法 -> 旧推荐清空
对手回应 -> 使用完整 moves 再分析
恢复到历史快照 -> moves 截断且版本递增
未知恢复棋盘 -> 发出 needs_sync，不猜测行棋方
```

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.LoopTests`

Expected: FAIL，旧 `run_loop` 仍使用 `TurnTracker` 和 FEN 快照。

- [x] **Step 3: 改造循环**

核心顺序必须是：

```python
result = motion.observe(frame, state.side_to_move)
if result.recovery:
    if state.restore_snapshot(result.board):
        motion.reset(frame, state.board)
    else:
        self.results.put(("needs_sync", result.board))
        self.stop_event.set()
        continue
elif result.board != state.board:
    if state.apply_board(result.board) and state.side_to_move != state.player:
        self.results.put(("clear_move", "等待对手落子"))

if state.needs_analysis():
    move, score = self.best_move(state.position_command(), movetime)
    state.mark_analyzed()
```

保留现有队列、线程、超时和引擎重启，不引入新的后台线程。

- [x] **Step 4: 运行循环测试**

Run: `python3 -m unittest -v test_assistant.LoopTests test_assistant.TurnTests`

Expected: PASS；完成后删除不再使用的 `TurnTracker` 及其测试，或仅在仍有调用者时保留。

### Task 6: 任意中盘、FEN 同步和搜索时间 UI

**Files:**
- Modify: `app.py`
- Modify: `README.md`
- Test: `test_assistant.py`

**Interfaces:**
- Produces: `parse_movetime_seconds(value: str) -> int`
- UI adds: `side_to_move_var`、`movetime_var`、`sync_fen()`。

- [x] **Step 1: 写输入失败测试**

```python
self.assertEqual(parse_movetime_seconds("5"), 5000)
for value in ("0", "61", "x", ""):
    with self.assertRaises(ValueError):
        parse_movetime_seconds(value)
```

另写 FEN 同步测试，断言 resync 后旧 moves 清空、active color 来自 FEN、版本递增。

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.LoopTests.test_movetime_accepts_only_one_to_sixty_seconds`

Expected: FAIL，解析函数不存在。

- [x] **Step 3: 实现最小 UI**

```python
def parse_movetime_seconds(value):
    try:
        seconds = int(value)
    except ValueError as error:
        raise ValueError("搜索秒数必须是1到60的整数") from error
    if not 1 <= seconds <= 60:
        raise ValueError("搜索秒数必须是1到60的整数")
    return seconds * 1000
```

增加“当前行棋”只读下拉、“搜索秒数”Spinbox 和“同步 FEN”按钮。`sync_fen()` 使用 `tkinter.simpledialog.askstring`，解析成功后保存待启动基点；失败时不得覆盖原状态。

启动辅助时：标准棋盘强制红方先行；非标准但有效棋盘使用“当前行棋”；识别失败时要求 FEN 同步。

- [x] **Step 4: 更新文档和运行 UI 测试**

README 写清一次标准校准、任意中盘、FEN、悔棋、失步和 5 秒搜索语义。

Run: `python3 -m unittest -v test_assistant.LoopTests`

Expected: PASS。

### Task 7: 完整验证与范围复查

**Files:**
- Test: `test_assistant.py`
- Review: `app.py`, `board_recognition.py`, `game_state.py`, `pikafish_engine.py`, `README.md`

- [x] **Step 1: 运行完整测试**

Run: `python3 -m unittest -v test_assistant.py`

Expected: 0 failures、0 errors；缺少真实腾讯截图附件时允许 1 skip。

- [x] **Step 2: 运行语法检查**

Run: `python3 -m py_compile app.py capture.py board_recognition.py game_state.py pikafish_engine.py test_assistant.py`

Expected: exit code 0。

- [x] **Step 3: 运行真实引擎历史验证**

从标准开局应用不同于初始推荐的红黑两步，断言 `position_command()` 包含两步 UCI，Pikafish 在 5 秒预算内返回合法 UCI 着法。

- [x] **Step 4: 计划覆盖复查**

确认设计中的完整历史、版本、重复局面、悔棋、两步恢复、三帧失步、任意中盘、FEN、1～60 秒预算、旧提示清空和 Windows 无控制台均有自动测试或明确人工验收项。

- [x] **Step 5: Windows 人工验收说明**

重新运行 `build_exe.bat`，测试标准开局、非推荐走法、中盘 FEN、单步/多步悔棋、重复局面、模拟失步、搜索秒数和 Pikafish 无黑框。当前非 Windows 环境只报告该项待实机确认，不声称已经通过。
