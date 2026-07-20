# Manual Sync Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make “同步并分析” capture exactly one current board, save that snapshot, analyze it once for the user's side, display the move, and then remain idle.

**Architecture:** Keep calibration, full-board recognition, FEN conversion, Pikafish, and the existing Tk result queue. Replace the main UI entrypoint with a one-shot worker; do not route it through `run_loop` or `MotionTracker`. Keep the old tracker implementation untouched because it remains independently tested and deleting it is outside this change.

**Tech Stack:** Python 3.10, tkinter/ttk, threading, OpenCV, Pillow, unittest.

## Global Constraints

- “同步并分析” is the only main-window analysis entrypoint.
- One click captures one frame and invokes Pikafish once.
- The analyzed side always equals “我执”.
- No polling or autonomous opponent-move detection occurs after analysis.
- Save the exact analyzed frame as `analysis_snapshot.png`.
- Do not add dependencies.
- The directory is not a Git repository, so commit steps are omitted.

---

### Task 1: Analysis snapshot and diagnostic preview

**Files:**
- Modify: `app.py:37-80`
- Test: `test_assistant.py:840-875`

**Interfaces:**
- Produces: `save_analysis_snapshot(directory: PathLike, frame: ndarray) -> None`
- Produces: `load_diagnostic_images(directory, max_size) -> list[dict]` whose main item is `analysis_snapshot.png`

- [ ] **Step 1: Write failing tests**

Add tests that save a frame, assert `analysis_snapshot.png` exists, and assert the preview title is “最近一次同步分析快照”.

```python
def test_analysis_snapshot_overwrites_latest_frame(self):
    first = np.zeros((10, 10, 3), dtype=np.uint8)
    second = np.full((10, 10, 3), 255, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        app_module.save_analysis_snapshot(Path(tmp), first)
        app_module.save_analysis_snapshot(Path(tmp), second)
        saved = cv2.imread(str(Path(tmp) / "analysis_snapshot.png"))
    self.assertTrue(np.array_equal(saved, second))

def test_diagnostic_preview_uses_latest_analysis_snapshot(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "analysis_snapshot.png"
        cv2.imwrite(str(path), np.zeros((100, 200, 3), dtype=np.uint8))
        items = app_module.load_diagnostic_images(Path(tmp), (40, 40))
    self.assertEqual(len(items), 1)
    self.assertEqual(items[0]["title"], "最近一次同步分析快照")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m unittest \
  test_assistant.LoopTests.test_analysis_snapshot_overwrites_latest_frame \
  test_assistant.LoopTests.test_diagnostic_preview_uses_latest_analysis_snapshot
```

Expected: failure because `save_analysis_snapshot` does not exist and the old preview has two recovery images.

- [ ] **Step 3: Implement the minimum snapshot path**

Add:

```python
def save_analysis_snapshot(directory, frame):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "analysis_snapshot.png"
    if frame is None or not cv2.imwrite(str(path), frame):
        raise OSError(f"无法保存分析快照: {path}")
```

Change `load_diagnostic_images` to load only:

```python
(("最近一次同步分析快照", "analysis_snapshot.png"),)
```

Update the empty-preview explanatory text to say the image is created by clicking “同步并分析”.

- [ ] **Step 4: Run the two tests and verify GREEN**

Expected: two tests pass.

---

### Task 2: One-shot engine worker

**Files:**
- Modify: `app.py` inside `AssistantApp`
- Test: `test_assistant.py` inside `LoopTests`

**Interfaces:**
- Consumes: `best_move(position: str, movetime: int) -> tuple[str | None, str]`
- Produces: `analyze_snapshot(board: tuple, position: str, movetime: int) -> None`
- Publishes: `("move", chinese_move, engine_detail)`, `("error", message)`, and finally `("stopped",)`

- [ ] **Step 1: Write failing one-shot tests**

Add a direct worker test:

```python
def test_analyze_snapshot_calls_engine_once_and_stops(self):
    app = object.__new__(AssistantApp)
    app.stop_event = threading.Event()
    app.results = Queue()
    app.best_move = Mock(return_value=("b2e2", "+0.20"))

    app.analyze_snapshot(
        STANDARD_BOARD,
        board_to_fen(STANDARD_BOARD, "w"),
        1000,
    )

    app.best_move.assert_called_once()
    events = list(app.results.queue)
    self.assertEqual(events[0][0], "move")
    self.assertEqual(events[-1], ("stopped",))
```

Add a cancellation test that sets `stop_event` during `best_move` and asserts that no `move` event is published.

- [ ] **Step 2: Run tests and verify RED**

Expected: `AttributeError` because `analyze_snapshot` is absent.

- [ ] **Step 3: Implement one-shot analysis**

```python
def analyze_snapshot(self, board, position, movetime):
    try:
        move, score = self.best_move(position, movetime)
        if self.stop_event.is_set():
            return
        if not move:
            raise RuntimeError("Pikafish未返回着法")
        text = PikafishEngine.get_chinese_move(move, board)
        self.results.put(("move", text, f"{move} · {score}"))
    except Exception as error:
        if not self.stop_event.is_set():
            self.results.put(("error", str(error)))
    finally:
        self.results.put(("stopped",))
```

- [ ] **Step 4: Run the worker tests and verify GREEN**

Expected: worker and cancellation tests pass.

---

### Task 3: Make “同步并分析” the only UI entrypoint

**Files:**
- Modify: `app.py:129-190, 380-408, 473-476, 579-610`
- Test: replace obsolete synchronization expectations in `test_assistant.py:1038-1100`

**Interfaces:**
- Consumes: `capture()`, `Calibration.recognize(frame)`, `save_analysis_snapshot`, `board_to_fen`, and `analyze_snapshot`
- Produces: `sync_current_board() -> None`, which starts exactly one worker

- [ ] **Step 1: Write failing click-flow tests**

Cover:

```python
@patch("app.threading.Thread")
@patch("app.save_analysis_snapshot")
def test_sync_current_board_captures_saves_and_starts_analysis(
    self, save_snapshot, thread_class
):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    app = object.__new__(AssistantApp)
    app.worker = None
    app.calibration = Mock(rotated=False, source_id="")
    app.calibration.recognize.return_value = br.Recognition(
        STANDARD_BOARD, 1.0, True
    )
    app.source_var = Mock()
    app.source_var.get.return_value = "window:test"
    app.sources = {"window:test": object()}
    app.source_id = Mock(return_value="window:test")
    app.player_var = Mock()
    app.player_var.get.return_value = "红方"
    app.movetime_var = Mock()
    app.movetime_var.get.return_value = "1"
    app.capture = Mock(return_value=frame)
    app.config_dir = Path("diagnostics")
    app.stop_event = threading.Event()
    app.move_var = Mock()
    app.status_var = Mock()
    app.side_to_move_var = Mock()
    app.sync_button = Mock()
    app.start_button = Mock()
    app.player_box = Mock()
    app.side_to_move_box = Mock()

    app.sync_current_board()

    app.capture.assert_called_once_with()
    app.calibration.recognize.assert_called_once_with(frame)
    save_snapshot.assert_called_once_with(app.config_dir, frame)
    kwargs = thread_class.call_args.kwargs
    self.assertIs(kwargs["target"].__self__, app)
    self.assertEqual(kwargs["target"].__func__, app.analyze_snapshot.__func__)
    self.assertEqual(kwargs["args"][1], board_to_fen(STANDARD_BOARD, "w"))
    self.assertEqual(kwargs["args"][2], 1000)
    thread_class.return_value.start.assert_called_once_with()
```

Repeat this setup with “黑方” and assert the FEN active side is `b`; return
`Recognition(valid=False)` and assert `Thread` is not constructed; set
`worker.is_alive()` true and assert `capture` is not called.

- [ ] **Step 2: Run click-flow tests and verify RED**

Expected: current method either refuses the active case or only assigns `synced_position`; it never starts the one-shot worker.

- [ ] **Step 3: Rewire the controls**

In `__init__`:

- Store the “同步并分析” button as `self.sync_button`.
- Set its command to `sync_current_board`.
- Change `start_button` to “停止分析”, command `stop`, initially disabled.
- Change “局面轮到” label to “本次分析方”.
- Disable direct editing of `side_to_move_box`.

Replace `sync_current_board` with this sequence:

```python
if self.worker and self.worker.is_alive():
    self.status_var.set("正在分析，请稍候或点击“停止分析”")
    return
if not self.calibration:
    raise ValueError("请先在标准开局校准棋盘")
player = "w" if self.player_var.get() == "红方" else "b"
if self.calibration.rotated != (player == "b"):
    raise ValueError("执棋方已改变，请重新校准棋盘")
if self.calibration.source_id and self.calibration.source_id != self.source_id():
    raise ValueError("模拟器采集源已改变，请重新校准棋盘")
if self.source_var.get() not in self.sources:
    raise ValueError("请选择可用模拟器")
movetime = parse_movetime_seconds(self.movetime_var.get())
self.status_var.set("正在截取当前棋盘")
frame = self.capture()
save_analysis_snapshot(self.config_dir, frame)
recognized = self.calibration.recognize(frame)
if not recognized.valid:
    raise ValueError(f"无法识别当前棋盘，请重新校准: {recognized.error}")
position = board_to_fen(recognized.board, player)
self.move_var.set("--")
self.side_to_move_var.set("红方" if player == "w" else "黑方")
self.stop_event.clear()
self.worker = threading.Thread(
    target=self.analyze_snapshot,
    args=(recognized.board, position, movetime),
    daemon=True,
)
self.worker.start()
self.sync_button.configure(state="disabled")
self.start_button.configure(state="normal")
self.player_box.configure(state="disabled")
self.status_var.set("已保存快照，正在分析")
```

Do not call `toggle`, `run_loop`, or `MotionTracker`.

- [ ] **Step 4: Update stop and result handling**

`stop()` sets the event, disables the stop button, and displays “正在停止分析”. It does not re-enable synchronization until the worker emits `("stopped",)`.

For `("stopped",)`, `poll_results`:

```python
self.sync_button.configure(state="normal")
self.start_button.configure(state="disabled")
self.player_box.configure(state="readonly")
self.side_to_move_box.configure(state="disabled")
```

It must not overwrite a successfully displayed move or its score.

- [ ] **Step 5: Run click-flow and UI-result tests**

Expected: all new tests pass.

---

### Task 4: Regression and documentation verification

**Files:**
- Modify: `README.md`
- Test: all `test_assistant.py`

**Interfaces:**
- No new runtime interfaces.

- [ ] **Step 1: Update usage documentation**

Describe the final workflow:

```text
校准棋盘 → 选择我执 → 等棋盘稳定 → 点击“同步并分析”
→ 查看保存的快照 → 获取一步建议 → 下次需要时再次点击
```

Remove instructions that tell users to start continuous monitoring or wait for automatic opponent detection.

- [ ] **Step 2: Run focused tests**

```bash
python3 -m unittest \
  test_assistant.LoopTests.test_analysis_snapshot_overwrites_latest_frame \
  test_assistant.LoopTests.test_analyze_snapshot_calls_engine_once_and_stops \
  test_assistant.LoopTests.test_sync_current_board_captures_saves_and_starts_analysis
```

Expected: PASS.

- [ ] **Step 3: Run the complete suite**

```bash
python3 -m unittest -v
```

Expected: all runnable tests pass; the existing external Tencent screenshot test may remain skipped when its fixture is absent.

- [ ] **Step 4: Compile-check modified Python files**

```bash
python3 -m py_compile app.py test_assistant.py
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Verify requirement boundaries**

Confirm from tests and code:

- One click produces one capture and one search.
- Active side equals the player.
- No call path from “同步并分析” reaches `run_loop`.
- The result persists after worker completion.
- The preview displays `analysis_snapshot.png`.
- No dependency was added.
