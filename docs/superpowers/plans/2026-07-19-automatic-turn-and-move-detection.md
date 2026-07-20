# 自动回合与落子识别 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 可靠区分“选中棋子”和“完成落子”，自动维护并显示真实回合，在标准开局校准时自动判断红黑视角，并为无法识别的画面留下有限诊断证据。

**Architecture:** 棋盘90个交叉点、象棋合法性和双帧稳定仍是唯一状态转换主链。`MotionTracker` 增加原因与变化证据，`GameState.side_to_move` 通过线程事件驱动 UI；校准后用棋子字色推断视角，无法推断时退回现有人工选择。GUI 外围动画不直接驱动棋局。

**Tech Stack:** Python 3、Tkinter、OpenCV、NumPy、现有 `unittest` 测试套件、Pikafish UCI。

## Global Constraints

- 选中光圈但棋子未离开原交叉点时不得产生走法。
- 唯一合法走法必须连续稳定两帧才提交。
- `GameState.side_to_move` 是运行期唯一真实回合状态。
- 自动视角判断只在标准开局校准时使用；不确定时安全退回人工选择。
- GUI 外围信号不得覆盖棋盘合法性结果。
- 诊断截图只保留最新一组，不无限增长。
- 不新增依赖、模型、OCR、联网服务或新的状态管理层。
- 项目目录不是 Git 仓库，因此任务不包含无法执行的提交步骤。

## File Structure

- Modify: `board_recognition.py` — 选择/走法原因、变化证据、红黑字色视角推断。
- Modify: `app.py` — 实时回合事件、控件状态、诊断帧保存、自动视角应用。
- Modify: `test_assistant.py` — 高亮组合、实时回合、视角判断、诊断文件测试。
- Modify: `README.md` — 说明自动状态、选中与落子的区别、诊断文件位置。
- No new runtime modules or dependencies.

---

### Task 1: 锁定选中、高亮与真实落子的行为

**Files:**
- Modify: `test_assistant.py:215-250`
- Modify only if the new test fails for the intended reason: `board_recognition.py:486-646`

**Interfaces:**
- Consumes: `MotionTracker.observe(frame, side) -> Recognition`
- Preserves: accepted board changes only after two identical legal candidates.

- [ ] **Step 1: Add a shared synthetic cell copier**

在测试方法内部继续使用现有局部辅助函数，不创建生产抽象：

```python
def copy_cell(target, base, source, destination):
    sx, sy = points[source[0]][source[1]]
    dx, dy = points[destination[0]][destination[1]]
    target[dy - half : dy - half + size, dx - half : dx - half + size] = (
        base[sy - half : sy - half + size, sx - half : sx - half + size]
    )
```

- [ ] **Step 2: Write the failing/characterization test for selection only**

```python
def test_motion_does_not_accept_selected_piece_without_displacement(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    points = grid_points(rect)
    selected = image.copy()
    cv2.circle(selected, points[3][4], 14, (0, 255, 0), 3)
    tracker = MotionTracker(calibration)
    tracker.observe(image, "b")

    first = tracker.observe(selected, "b")
    second = tracker.observe(selected, "b")

    self.assertEqual(first.board, STANDARD_BOARD)
    self.assertEqual(second.board, STANDARD_BOARD)
    self.assertFalse(first.recovery)
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.RecognitionTests.test_motion_does_not_accept_selected_piece_without_displacement -v
```

Expected: 若现有不变量正确则 `OK`；这是安全特征测试，立即通过时不改生产代码。

- [ ] **Step 3: Write the high-noise opponent-move test**

构造“红方上一手高亮消失 + 黑卒前进 + 黑方新高亮出现”同帧场景：

```python
def test_motion_accepts_opponent_move_when_highlights_switch_in_same_frame(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    points = grid_points(rect)
    size, half = calibration.size, calibration.size // 2

    def copy_cell(target, base, source, destination):
        sx, sy = points[source[0]][source[1]]
        dx, dy = points[destination[0]][destination[1]]
        target[dy - half : dy - half + size, dx - half : dx - half + size] = (
            base[sy - half : sy - half + size, sx - half : sx - half + size]
        )

    after_red = image.copy()
    copy_cell(after_red, image, (5, 4), (7, 1))
    copy_cell(after_red, image, (7, 1), (7, 4))
    red_highlight = after_red.copy()
    cv2.circle(red_highlight, points[7][1], 12, (0, 255, 0), 3)
    cv2.circle(red_highlight, points[7][4], 12, (0, 255, 0), 3)

    tracker = MotionTracker(calibration)
    tracker.observe(image, "w")
    tracker.observe(red_highlight, "w")
    accepted_red = tracker.observe(red_highlight, "w")

    after_black = after_red.copy()
    copy_cell(after_black, image, (5, 4), (3, 4))
    copy_cell(after_black, image, (3, 4), (4, 4))
    black_highlight = after_black.copy()
    cv2.circle(black_highlight, points[3][4], 12, (0, 255, 0), 3)
    cv2.circle(black_highlight, points[4][4], 12, (0, 255, 0), 3)

    tracker.observe(black_highlight, "b")
    accepted_black = tracker.observe(black_highlight, "b")

    self.assertEqual(
        accepted_black.board,
        moved(accepted_red.board, (3, 4), (4, 4)),
    )
```

- [ ] **Step 4: Run the test and classify the result**

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.RecognitionTests.test_motion_accepts_opponent_move_when_highlights_switch_in_same_frame -v
```

Expected: `OK` means existing legal-candidate algorithm already handles the pattern and no production patch is allowed. If it fails because one move endpoint falls outside the six retained changes, make the single minimal change below and re-run:

```python
return [item for item in scores if item[1] >= threshold][:8]
```

Do not lower the `0.04` threshold; that would increase global noise and is outside the demonstrated failure.

- [ ] **Step 5: Run all recognition tests**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.RecognitionTests -v
```

Expected: all available recognition tests pass; the existing external Tencent sample may remain skipped when absent.

---

### Task 2: 让真实回合状态驱动 UI

**Files:**
- Modify: `test_assistant.py:490-530,696-820`
- Modify: `app.py:90-120,255-342,344-400,423-450`

**Interfaces:**
- Produces queue event: `("turn", side: Literal["w", "b"])`
- Consumes: `GameState.side_to_move`
- Stores widgets: `self.player_box`, `self.side_to_move_box`

- [ ] **Step 1: Write a failing run-loop turn-event test**

在现有红黑连续走法的 `run_loop` 测试末尾增加：

```python
turns = [event[1] for event in events if event[0] == "turn"]
self.assertEqual(turns, ["w", "b", "w"])
```

Run the containing test by its existing test name shown by:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.RecognitionTests.test_recognized_red_and_black_moves_trigger_second_advice -v
```

Expected: `FAIL` because no `turn` events exist.

- [ ] **Step 2: Emit turn events at every proven state transition**

在 `run_loop()` 中加入以下事件，不直接从工作线程操作 Tkinter：

```python
if state is not None:
    self.results.put(("turn", state.side_to_move))
```

首次由识别结果创建状态后：

```python
state = GameState.start(result.board, "w", player)
self.results.put(("turn", state.side_to_move))
```

恢复历史快照成功后：

```python
self.results.put(("turn", state.side_to_move))
```

`state.apply_board(result.board)` 成功后：

```python
self.results.put(("turn", state.side_to_move))
```

初始传入 `state` 的事件只能发送一次，不能在每个循环重复发送。

- [ ] **Step 3: Run the focused run-loop test**

Run the Step 1 command.

Expected: `OK` with exactly `w -> b -> w`.

- [ ] **Step 4: Write a failing poll-results UI test**

```python
def test_poll_results_updates_live_side_to_move(self):
    app = object.__new__(AssistantApp)
    app.results = Queue()
    app.results.put(("turn", "b"))
    app.side_to_move_var = Mock()
    app.move_var = Mock()
    app.status_var = Mock()
    app.start_button = Mock()
    app.root = Mock()

    app.poll_results()

    app.side_to_move_var.set.assert_called_once_with("黑方")
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_poll_results_updates_live_side_to_move -v
```

Expected: `FAIL` because `turn` is ignored.

- [ ] **Step 5: Update widgets and UI event handling**

Store the existing comboboxes instead of discarding their references:

```python
self.player_box = ttk.Combobox(
    frame,
    textvariable=self.player_var,
    state="readonly",
    values=("红方", "黑方"),
    width=8,
)
self.player_box.grid(row=1, column=1, sticky="w", padx=(8, 0))
```

```python
ttk.Label(frame, text="局面轮到").grid(row=2, column=0, sticky="w")
self.side_to_move_box = ttk.Combobox(
    frame,
    textvariable=self.side_to_move_var,
    state="readonly",
    values=("红方", "黑方"),
    width=8,
)
self.side_to_move_box.grid(row=2, column=1, sticky="w", padx=(8, 0))
```

成功启动线程时锁定两个控件：

```python
self.player_box.configure(state="disabled")
self.side_to_move_box.configure(state="disabled")
```

`poll_results()` 增加：

```python
elif event[0] == "turn":
    self.side_to_move_var.set("红方" if event[1] == "w" else "黑方")
```

收到 `needs_sync` 或 `stopped` 时恢复 `side_to_move_box` 为 `readonly`；收到 `stopped` 时同时恢复 `player_box`。`sync_current_board()` 成功后按选定 side 更新显示，无需另建状态。

- [ ] **Step 6: Run all loop tests**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests -v
```

Expected: all `LoopTests` pass.

---

### Task 3: 标准开局自动判断用户执棋方

**Files:**
- Modify: `test_assistant.py:110-140,530-580`
- Modify: `board_recognition.py:34-115`
- Modify: `app.py:12-18,187-210`

**Interfaces:**
- Produces: `infer_player_from_colors(image, rect) -> str | None`, returning `"w"`, `"b"`, or `None`.
- Consumes: BGR screenshot and board rectangle `(left, top, right, bottom)`.

- [ ] **Step 1: Write failing orientation tests**

Import `infer_player_from_colors` from `board_recognition`, then add:

```python
def test_piece_colors_infer_red_and_black_player_views(self):
    image, rect = standard_board_image()
    self.assertEqual(infer_player_from_colors(image, rect), "w")

    rotated = cv2.rotate(image, cv2.ROTATE_180)
    height, width = image.shape[:2]
    rotated_rect = (
        width - rect[2],
        height - rect[3],
        width - rect[0],
        height - rect[1],
    )
    self.assertEqual(infer_player_from_colors(rotated, rotated_rect), "b")
```

```python
def test_piece_colors_return_none_when_red_signal_is_ambiguous(self):
    image, rect = standard_board_image()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    self.assertIsNone(infer_player_from_colors(gray, rect))
```

- [ ] **Step 2: Run tests to verify RED**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest \
  test_assistant.BoardTests.test_piece_colors_infer_red_and_black_player_views \
  test_assistant.BoardTests.test_piece_colors_return_none_when_red_signal_is_ambiguous -v
```

Expected: import error or missing-function error.

- [ ] **Step 3: Implement the minimal HSV red-ink comparison**

Add beside `grid_points()`:

```python
def infer_player_from_colors(image, rect):
    if image is None:
        return None
    points = grid_points(rect)
    dx = abs(points[0][1][0] - points[0][0][0])
    dy = abs(points[1][0][1] - points[0][0][1])
    radius = max(6, int(min(dx, dy) * 0.22))

    def red_score(row):
        red_pixels = pixels = 0
        for x, y in points[row]:
            crop = image[y - radius : y + radius, x - radius : x + radius]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            red = ((hue < 12) | (hue > 170)) & (saturation > 100) & (value > 70)
            red_pixels += int(np.count_nonzero(red))
            pixels += red.size
        return red_pixels / pixels if pixels else 0.0

    top, bottom = red_score(0), red_score(9)
    if bottom > 0.005 and bottom > top * 1.5:
        return "w"
    if top > 0.005 and top > bottom * 1.5:
        return "b"
    return None
```

- [ ] **Step 4: Run orientation tests to verify GREEN**

Run the Step 2 command.

Expected: both tests `OK`.

- [ ] **Step 5: Apply automatic inference after board calibration**

Import the helper in `app.py`. Keep `create_calibration()` unchanged. In `calibrate()` after the existing calibration succeeds:

```python
detected_player = infer_player_from_colors(frame, calibration.rect)
if detected_player is not None:
    rotated = detected_player == "b"
    if calibration.rotated != rotated:
        calibration = Calibration.create(
            frame, calibration.rect, rotated, self.source_id()
        )
    self.player_var.set("红方" if detected_player == "w" else "黑方")
else:
    detected_player = "b" if calibration.rotated else "w"
```

Save the final calibration after possible recreation. Set status to:

```python
message = f"{mode}校准成功，已识别我执{'红方' if detected_player == 'w' else '黑方'}"
```

For ambiguous colors append `"（使用人工选择）"`; implement this with a boolean `automatic_player = detected_player is not None` captured before fallback.

When loading persisted calibration in `__init__`, set:

```python
self.player_var.set("黑方" if self.calibration.rotated else "红方")
```

- [ ] **Step 6: Run board and calibration tests**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.BoardTests test_assistant.RecognitionTests -v
```

Expected: all available tests pass; only the existing absent Tencent sample may skip.

---

### Task 4: 增加原因分类和有限诊断截图

**Files:**
- Modify: `test_assistant.py:250-330,696-820`
- Modify: `board_recognition.py:198-205,506-646`
- Modify: `app.py:30-58,344-400`
- Modify: `README.md:18-95`

**Interfaces:**
- Extends: `Recognition.reason: str = ""`
- Extends: `Recognition.changes: tuple = ()`
- Produces: `save_diagnostic_frames(directory, reference, current) -> None`

- [ ] **Step 1: Write failing reason/evidence tests**

Extend the selection-only test with:

```python
self.assertEqual(first.reason, "no_move")
self.assertEqual(len(first.changes), 1)
self.assertEqual(first.error, "检测到选中操作，等待棋子落点")
```

Extend the existing recovery-failure test with:

```python
self.assertEqual(result.reason, "recovery_failed")
self.assertGreaterEqual(len(result.changes), 2)
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest \
  test_assistant.RecognitionTests.test_motion_does_not_accept_selected_piece_without_displacement \
  test_assistant.RecognitionTests.test_motion_preserves_full_recognition_failure_reason -v
```

Expected: missing attributes or assertion failures.

- [ ] **Step 2: Return candidate reasons without changing acceptance rules**

Extend `Recognition` after the existing `recovery` field:

```python
reason: str = ""
changes: tuple = ()
```

Change `_infer_move_sequence_from_changes()` to return `(board_sequence, reason)`:

```python
if len(changed) < 2:
    return None, "no_move"
# existing sequence construction
if not sequences:
    return None, "illegal"
# existing sorting
if ambiguous:
    return None, "ambiguous"
return best[2], ""
```

Update its sole production caller:

```python
proposed, reason = _infer_move_sequence_from_changes(self.board, changed, side)
changes = tuple(changed)
```

For one changed intersection return:

```python
error = "检测到选中操作，等待棋子落点" if len(changed) == 1 else existing_error
return Recognition(
    self.board, 1.0, True, error,
    reason=reason, changes=changes,
)
```

For the first candidate frame return `reason="unstable"`; for full recognition failure return `reason="recovery_failed"`. Preserve `recovery=True` for successful full-board recovery.

- [ ] **Step 3: Run reason tests to verify GREEN**

Run the Step 1 command.

Expected: both tests `OK`.

- [ ] **Step 4: Write failing finite-diagnostic-file test**

Import `save_diagnostic_frames` from `app`, then add:

```python
def test_diagnostic_frames_overwrite_one_latest_pair(self):
    first = np.zeros((10, 10, 3), dtype=np.uint8)
    second = np.full((10, 10, 3), 255, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        save_diagnostic_frames(Path(tmp), first, second)
        save_diagnostic_frames(Path(tmp), second, first)
        files = sorted(path.name for path in Path(tmp).glob("diagnostic_*.png"))
    self.assertEqual(files, ["diagnostic_current.png", "diagnostic_reference.png"])
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_diagnostic_frames_overwrite_one_latest_pair -v
```

Expected: import error or missing-function error.

- [ ] **Step 5: Implement and call finite diagnostic saving**

Add in `app.py`:

```python
def save_diagnostic_frames(directory, reference, current):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = {
        "diagnostic_reference.png": reference,
        "diagnostic_current.png": current,
    }
    for name, image in images.items():
        if image is None or not cv2.imwrite(str(directory / name), image):
            raise OSError(f"无法保存诊断截图: {name}")
```

In `run_loop()`, immediately after `motion.observe()`:

```python
diagnostic_error = ""
if result.reason == "recovery_failed":
    try:
        save_diagnostic_frames(self.config_dir, motion.reference, frame)
    except OSError as error:
        diagnostic_error = f"；{error}"
```

When publishing `result.error`, append `diagnostic_error`. Do not stop the game or mutate `GameState` because diagnostic saving failed.

- [ ] **Step 6: Update README user guidance**

Document these exact behaviors:

- “局面轮到” is live state and is only manually selected for unknown-position recovery.
- “我执” is inferred during standard calibration when red/black ink is distinguishable.
- A green ring without coordinate displacement means selection, not a completed move.
- Latest recovery evidence is stored as `diagnostic_reference.png` and `diagnostic_current.png` under `%APPDATA%\XiangqiAssistant`, overwriting the prior pair.
- GUI animations are auxiliary evidence; board legality remains authoritative.

- [ ] **Step 7: Run literal UX scan**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
rg -n "当前行棋|同步 ?FEN|请输入.*FEN|FEN无效" app.py board_recognition.py README.md
```

Expected: no old `当前行棋` label and no user-facing FEN operation text. Internal FEN functions/tests may remain.

---

### Task 5: Complete verification

**Files:**
- Verify: `app.py`, `board_recognition.py`, `game_state.py`, `pikafish_engine.py`, `test_assistant.py`, `README.md`

**Interfaces:**
- No new behavior; verification only.

- [ ] **Step 1: Run the complete suite**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant -v
```

Expected: all available tests pass; only the existing missing external Tencent screenshot test may skip.

- [ ] **Step 2: Compile all changed Python modules**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m py_compile app.py board_recognition.py game_state.py pikafish_engine.py test_assistant.py
```

Expected: exit code `0`, no output.

- [ ] **Step 3: Run the real engine smoke test**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.EngineTests.test_real_engine_returns_legal_uci_move -v
```

Expected: `OK` with a legal UCI move.

- [ ] **Step 4: Record platform boundary**

The macOS environment cannot validate Windows Tkinter rendering, JJ window capture, or the rebuilt EXE. Rebuild with `build_exe.bat` on Windows and manually verify: selection only, red cannon move, black pawn move, highlight switch, live turn display, red/black player views, recovery diagnostics, and hidden Pikafish console. Do not report those checks as passed until actually run on Windows.
