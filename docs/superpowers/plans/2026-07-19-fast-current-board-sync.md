# 快速响应与当前棋盘同步 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将默认落子提示缩短到约 2 秒内，并用“自动识别棋盘 + 用户选择当前行棋方”替代面向用户的 FEN 输入。

**Architecture:** 只修改现有 `AssistantApp` 编排层：用两个模块常量统一默认搜索预算和采样周期；用 `sync_current_board()` 复用失步恢复已缓存的棋盘，或调用现有 `Calibration.recognize()` 识别当前截图。`GameState`、`MotionTracker` 与 Pikafish 协议保持不变，内部继续使用 FEN 作为引擎基点。

**Tech Stack:** Python 3、Tkinter、现有 OpenCV/Pillow 识别链路、`unittest`、Pikafish UCI。

## Global Constraints

- 默认搜索秒数必须为 `1`，用户仍可设置 `1～60` 秒。
- 棋盘采样周期必须为 `0.2` 秒，并保留连续两帧稳定确认。
- 普通 UI、提示和错误信息不得要求用户理解或输入 FEN。
- 静态棋盘无法可靠推断当前行棋方，未知局面由用户在现有下拉框选择红方或黑方。
- 不新增依赖、视觉模型、状态管理层或 JJ 象棋界面特效识别。
- 不改变完整历史、悔棋、重复局面和局面版本语义。
- 项目目录不是 Git 仓库，因此本计划不包含无法执行的提交步骤；所有变更直接保存在当前工作区。

## File Structure

- Modify: `app.py` — 默认预算、采样周期、同步当前棋盘入口与普通用户提示。
- Modify: `test_assistant.py` — 为延迟常量和同步逻辑增加最小回归测试。
- Modify: `README.md` — 更新默认响应时间和同步当前棋盘操作说明。
- No new runtime files or dependencies.

---

### Task 1: 统一快速响应参数

**Files:**
- Modify: `test_assistant.py:696`
- Modify: `app.py:30-52,82,326-368`

**Interfaces:**
- Produces: `DEFAULT_SEARCH_SECONDS: int = 1`
- Produces: `POLL_INTERVAL_SECONDS: float = 0.2`
- Consumes: existing `parse_movetime_seconds(value: str) -> int`

- [ ] **Step 1: Write the failing test**

在 `LoopTests` 中增加：

```python
def test_fast_defaults_use_one_second_search_and_point_two_sampling(self):
    self.assertEqual(app_module.DEFAULT_SEARCH_SECONDS, 1)
    self.assertEqual(app_module.POLL_INTERVAL_SECONDS, 0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_fast_defaults_use_one_second_search_and_point_two_sampling -v
```

Expected: `ERROR`，提示 `app` 没有 `DEFAULT_SEARCH_SECONDS`。

- [ ] **Step 3: Write minimal implementation**

在 `app.py` 导入区之后增加：

```python
DEFAULT_SEARCH_SECONDS = 1
POLL_INTERVAL_SECONDS = 0.2
```

将 Tkinter 默认值改为：

```python
self.movetime_var = tk.StringVar(value=str(DEFAULT_SEARCH_SECONDS))
```

将 `run_loop` 默认搜索预算及两处等待统一改为：

```python
def run_loop(
    self,
    player,
    state=None,
    initial_frame=None,
    movetime=DEFAULT_SEARCH_SECONDS * 1000,
):
```

```python
self.stop_event.wait(POLL_INTERVAL_SECONDS)
```

两处原有 `wait(0.7)` 都必须替换，其他运动识别和双帧确认代码不变。

- [ ] **Step 4: Run focused and loop tests**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests -v
```

Expected: `LoopTests` 全部 `OK`。

---

### Task 2: 用当前棋盘同步替代 FEN 弹窗

**Files:**
- Modify: `test_assistant.py:696`
- Modify: `app.py:7-18,127-129,255-319,399-415`

**Interfaces:**
- Produces: `AssistantApp.sync_current_board(self) -> None`
- Consumes: `self.pending_recovery_board: tuple | None`
- Consumes: `Calibration.recognize(frame) -> Recognition`
- Consumes: `board_to_fen(board, side: str) -> str`
- Preserves: `self.synced_position == (board, side_to_move, base_fen)` for the existing `toggle()` path.

- [ ] **Step 1: Write failing tests for cached-board synchronization**

在 `LoopTests` 增加：

```python
def test_sync_current_board_reuses_recovery_board_without_recognition(self):
    app = object.__new__(AssistantApp)
    app.worker = None
    app.pending_recovery_board = STANDARD_BOARD
    app.synced_position = None
    app.side_to_move_var = Mock()
    app.side_to_move_var.get.return_value = "黑方"
    app.move_var = Mock()
    app.status_var = Mock()
    app.capture = Mock(side_effect=AssertionError("不应重复截图"))
    app.calibration = Mock()

    app.sync_current_board()

    self.assertEqual(
        app.synced_position,
        (STANDARD_BOARD, "b", board_to_fen(STANDARD_BOARD, "b")),
    )
    self.assertIsNone(app.pending_recovery_board)
    app.capture.assert_not_called()
    app.calibration.recognize.assert_not_called()
    app.move_var.set.assert_called_once_with("--")
    app.status_var.set.assert_called_once_with("当前棋盘已同步，请开始辅助")
```

再增加无缓存识别成功测试：

```python
def test_sync_current_board_recognizes_frame_when_no_recovery_is_cached(self):
    app = object.__new__(AssistantApp)
    app.worker = None
    app.pending_recovery_board = None
    app.synced_position = None
    app.side_to_move_var = Mock()
    app.side_to_move_var.get.return_value = "红方"
    app.move_var = Mock()
    app.status_var = Mock()
    frame = object()
    app.capture = Mock(return_value=frame)
    app.calibration = Mock()
    app.calibration.recognize.return_value = br.Recognition(
        STANDARD_BOARD, 1.0, True
    )

    app.sync_current_board()

    app.calibration.recognize.assert_called_once_with(frame)
    self.assertEqual(
        app.synced_position,
        (STANDARD_BOARD, "w", board_to_fen(STANDARD_BOARD, "w")),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest \
  test_assistant.LoopTests.test_sync_current_board_reuses_recovery_board_without_recognition \
  test_assistant.LoopTests.test_sync_current_board_recognizes_frame_when_no_recovery_is_cached -v
```

Expected: 两项 `ERROR`，提示 `AssistantApp` 没有 `sync_current_board`。

- [ ] **Step 3: Write the minimal synchronization implementation**

在 `app.py`：

1. 从 Tkinter 导入中删除 `simpledialog`。
2. 从 `board_recognition` 导入中删除 `fen_to_board`，增加 `board_to_fen`。
3. 将按钮替换为：

```python
ttk.Button(
    frame, text="同步当前棋盘", command=self.sync_current_board
).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 4))
```

4. 用以下方法替换 `sync_fen()`：

```python
def sync_current_board(self):
    if self.worker and self.worker.is_alive():
        messagebox.showinfo("象棋辅助", "请先停止辅助")
        return
    try:
        board = self.pending_recovery_board
        if board is None:
            if not self.calibration:
                raise ValueError("请先在标准开局校准棋盘")
            recognized = self.calibration.recognize(self.capture())
            if not recognized.valid:
                raise ValueError(f"无法识别当前棋盘，请重新校准: {recognized.error}")
            board = recognized.board
        side = {"红方": "w", "黑方": "b"}[self.side_to_move_var.get()]
        fen = board_to_fen(board, side)
    except KeyError:
        messagebox.showerror("无法同步", "请选择当前行棋方：红方或黑方")
        self.status_var.set("请选择当前行棋方：红方或黑方")
        return
    except (OSError, RuntimeError, ValueError) as error:
        messagebox.showerror("无法同步", str(error))
        self.status_var.set(str(error))
        return
    self.synced_position = (board, side, fen)
    self.pending_recovery_board = None
    self.move_var.set("--")
    self.status_var.set("当前棋盘已同步，请开始辅助")
```

5. 将 `toggle()` 中识别失败文案改为：

```python
raise ValueError(f"无法识别当前棋盘，请重新校准: {recognized.error}")
```

6. 将 `poll_results()` 的 `needs_sync` 文案改为：

```python
self.status_var.set("请选择当前行棋方，然后点击“同步当前棋盘”")
```

- [ ] **Step 4: Run focused tests to verify they pass**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests -v
```

Expected: `LoopTests` 全部 `OK`。

- [ ] **Step 5: Add and verify the failure-path regression**

增加测试，确保识别失败不覆盖旧状态：

```python
@patch("app.messagebox.showerror")
def test_sync_current_board_keeps_state_when_recognition_fails(self, showerror):
    app = object.__new__(AssistantApp)
    app.worker = None
    app.pending_recovery_board = None
    original = (STANDARD_BOARD, "b", board_to_fen(STANDARD_BOARD, "b"))
    app.synced_position = original
    app.side_to_move_var = Mock()
    app.side_to_move_var.get.return_value = "红方"
    app.status_var = Mock()
    app.move_var = Mock()
    app.capture = Mock(return_value=object())
    app.calibration = Mock()
    app.calibration.recognize.return_value = br.Recognition(
        tuple(), 0.0, False, "置信度过低"
    )

    app.sync_current_board()

    self.assertEqual(app.synced_position, original)
    showerror.assert_called_once()
    app.status_var.set.assert_called_once_with(
        "无法识别当前棋盘，请重新校准: 置信度过低"
    )
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_sync_current_board_keeps_state_when_recognition_fails -v
```

Expected: `OK`。

---

### Task 3: 清理用户侧 FEN 文案并完成全量验证

**Files:**
- Modify: `README.md:18,42-46`
- Modify if needed: `board_recognition.py:615`
- Verify: `app.py`, `test_assistant.py`

**Interfaces:**
- No new runtime interfaces.
- Preserves internal `board_to_fen()` / `fen_to_board()` compatibility for Pikafish and existing tests.

- [ ] **Step 1: Remove remaining user-facing FEN instructions**

将 README 的操作说明改为以下事实：

```markdown
“搜索秒”默认是 1，可设置 1～60 秒；默认情况下通常约 2 秒内给出提示，增加搜索时间通常能提高棋力，但会等得更久。“当前行棋”用于从非标准中盘启动或失步恢复；静态棋盘无法包含轮到谁这一信息，因此请按实际情况选择红方或黑方。
```

```markdown
- 自动恢复到未知局面时，先选择“当前行棋”，再点击“同步当前棋盘”。程序负责识别棋子位置，不需要输入 FEN。
- 连续 3 帧无法解释时会尝试完整棋盘识别。未知局面无法从最终摆法反推出唯一旧历史，因此程序会从已识别局面建立新的历史基点。
- 默认搜索 1 秒，显示的是当前搜索预算内的最佳着法，不是数学意义上可证明的绝对最优解。
```

若 `board_recognition.py` 的恢复错误仍包含“请同步FEN”，改为：

```python
return Recognition(
    self.board, 0.0, True, "无法自动恢复，请同步当前棋盘"
)
```

- [ ] **Step 2: Verify no ordinary UI text exposes FEN**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
rg -n "同步 ?FEN|请输入.*FEN|FEN无效" app.py board_recognition.py README.md
```

Expected: 无输出。README 可以解释“不需要输入 FEN”，但不得提供 FEN 操作步骤；如该解释导致上述命令匹配，只检查匹配行确实是否定说明。

- [ ] **Step 3: Run the complete automated suite**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant -v
```

Expected: 所有可用测试通过；缺少仓库外截图资源 `123.jpg` 时仅允许原有对应测试跳过。

- [ ] **Step 4: Compile all changed Python modules**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m py_compile app.py board_recognition.py game_state.py pikafish_engine.py test_assistant.py
```

Expected: 退出码 `0`，无输出。

- [ ] **Step 5: Smoke-test the real engine after the UI budget change**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.EngineTests.test_real_engine_returns_legal_uci_move -v
```

Expected: `OK`，Pikafish 返回合法 UCI 着法。该现有测试使用 200ms 以缩短自动化耗时；UI 的 1 秒默认值由 Task 1 的常量测试保证。

- [ ] **Step 6: Record Windows manual acceptance scope**

重新构建 Windows EXE 后人工验证：默认 1 秒搜索、0.2 秒采样、红黑双方同步、未知局面恢复、界面无 FEN 输入框、Pikafish 无黑色控制台。当前 macOS 环境只记录该验收项，不虚构 Windows GUI 验证结果。
