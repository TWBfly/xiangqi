# 可视化诊断截图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在象棋辅助器内直接预览最近一次恢复尝试的前后画面，并可一键打开实际诊断目录。

**Architecture:** 保留两张固定 PNG 和现有保存函数；完整恢复成功或失败都触发覆盖保存。新增一个可独立测试的图片加载函数，Tkinter `Toplevel` 只负责展示加载结果；目录打开使用 Windows 原生 `os.startfile`，其他平台显示路径。

**Tech Stack:** Python 3、Tkinter、Pillow、OpenCV、stdlib `datetime`、现有 `unittest`。

## Global Constraints

- 诊断目录最多保留 `diagnostic_reference.png` 和 `diagnostic_current.png` 两张。
- 正常等待、单次选中和正常落子不保存。
- 完整恢复成功与失败都保存最新一组。
- 预览只读，不阻塞或修改 `GameState`。
- 无图片、单图缺失、图片损坏和目录打开失败必须显示中文信息。
- 不新增依赖、历史图库、自动弹窗或联网行为。
- 项目目录不是 Git 仓库，因此不包含提交步骤。

## File Structure

- Modify: `board_recognition.py` — 标记成功完整恢复原因。
- Modify: `app.py` — 图片加载、预览窗口、打开目录和按钮。
- Modify: `test_assistant.py` — 保存时机、图片加载和目录行为测试。
- Modify: `README.md` — 新按钮和无图状态说明。
- No new runtime files or dependencies.

---

### Task 1: 完整恢复成功时也保存诊断画面

**Files:**
- Modify: `test_assistant.py:360-410,827-870`
- Modify: `board_recognition.py:630-655`
- Modify: `app.py:390-405`

**Interfaces:**
- Produces: `Recognition.reason == "recovery_succeeded"` for successful full-board recovery.
- Preserves: `Recognition.reason == "recovery_failed"` for failed recovery.

- [ ] **Step 1: Write failing reason test**

在现有 `test_motion_requests_recovery_after_three_unresolved_frames` 中增加：

```python
self.assertEqual(result.reason, "recovery_succeeded")
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.RecognitionTests.test_motion_requests_recovery_after_three_unresolved_frames -v
```

Expected: `FAIL` because the reason is currently empty.

- [ ] **Step 2: Mark successful full recovery**

在成功完整识别返回值中加入：

```python
return Recognition(
    recovered.board,
    recovered.confidence,
    True,
    "已识别当前棋盘，正在恢复",
    True,
    reason="recovery_succeeded",
    changes=changes,
)
```

- [ ] **Step 3: Run focused test to verify GREEN**

Run the Step 1 command.

Expected: `OK`.

- [ ] **Step 4: Write failing successful-recovery save test**

保留现有失败恢复测试，新增独立测试：

```python
@patch("app.save_diagnostic_frames")
@patch("app.MotionTracker")
def test_run_loop_saves_frames_after_successful_recovery(
    self, tracker_class, save_frames
):
    from game_state import GameState

    class StopAfterWait:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _seconds):
            self.stopped = True
            return True

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    motion = tracker_class.return_value
    motion.reference = frame.copy()
    motion.observe.return_value = br.Recognition(
        STANDARD_BOARD,
        1.0,
        True,
        "恢复诊断",
        reason="recovery_succeeded",
    )
    app = object.__new__(AssistantApp)
    app.stop_event = StopAfterWait()
    app.results = Queue()
    app.engine_lock = threading.Lock()
    app.engine = None
    app.calibration = Mock()
    app.config_dir = Path("diagnostics")
    app.capture = Mock(return_value=frame)
    state = GameState.start(STANDARD_BOARD, "w", "b")

    app.run_loop("b", state, frame)

    save_frames.assert_called_once_with(
        app.config_dir, motion.reference, frame
    )
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_run_loop_saves_frames_after_successful_recovery -v
```

Expected: `FAIL` because `recovery_succeeded` does not yet save.

- [ ] **Step 5: Broaden the save condition minimally**

Replace:

```python
if result.reason == "recovery_failed":
```

with:

```python
if result.reason in {"recovery_succeeded", "recovery_failed"}:
```

Run the Step 4 command.

Expected: `OK`. The existing failure-recovery test must remain `OK`.

---

### Task 2: 可测试地加载诊断图片与状态

**Files:**
- Modify: `test_assistant.py:800-870`
- Modify: `app.py:1-48`

**Interfaces:**
- Produces: `load_diagnostic_images(directory, max_size=(400, 320)) -> list[dict]`
- Each dict has exact keys: `title`, `path`, `image`, `modified`, `error`.

- [ ] **Step 1: Write failing valid/missing/corrupt tests**

```python
def test_load_diagnostic_images_handles_valid_missing_and_corrupt_files(self):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        valid = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / "diagnostic_reference.png"), valid)
        items = app_module.load_diagnostic_images(directory, (40, 40))
        self.assertEqual(items[0]["title"], "参考画面")
        self.assertEqual(items[0]["image"].size, (40, 20))
        self.assertEqual(items[0]["error"], "")
        self.assertEqual(items[1]["error"], "图片不存在")

        (directory / "diagnostic_current.png").write_bytes(b"broken")
        items = app_module.load_diagnostic_images(directory, (40, 40))
        self.assertIn("无法读取图片", items[1]["error"])
        self.assertIsNone(items[1]["image"])
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests.test_load_diagnostic_images_handles_valid_missing_and_corrupt_files -v
```

Expected: missing-function error.

- [ ] **Step 2: Implement the minimal loader**

Import `datetime` and add:

```python
def load_diagnostic_images(directory, max_size=(400, 320)):
    directory = Path(directory)
    items = []
    for title, name in (
        ("参考画面", "diagnostic_reference.png"),
        ("当前画面", "diagnostic_current.png"),
    ):
        path = directory / name
        item = {
            "title": title,
            "path": path,
            "image": None,
            "modified": "",
            "error": "",
        }
        if not path.exists():
            item["error"] = "图片不存在"
        else:
            try:
                with Image.open(path) as source:
                    preview = source.copy()
                preview.thumbnail(max_size)
                item["image"] = preview
                item["modified"] = datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError) as error:
                item["error"] = f"无法读取图片: {error}"
        items.append(item)
    return items
```

- [ ] **Step 3: Run loader test to verify GREEN**

Run the Step 1 command.

Expected: `OK`.

---

### Task 3: 内置预览窗口和打开目录

**Files:**
- Modify: `test_assistant.py:800-900`
- Modify: `app.py:60-165,360-385`

**Interfaces:**
- Produces: `AssistantApp.show_diagnostics() -> None`
- Produces: `AssistantApp.open_diagnostic_directory() -> None`
- Consumes: `load_diagnostic_images(self.config_dir)`.

- [ ] **Step 1: Write failing directory-opening tests**

```python
@patch("app.messagebox.showinfo")
def test_open_diagnostic_directory_creates_and_shows_path_without_startfile(
    self, showinfo
):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "diagnostics"
        app = object.__new__(AssistantApp)
        app.config_dir = directory
        with patch.object(app_module.os, "startfile", None, create=True):
            app.open_diagnostic_directory()
        self.assertTrue(directory.is_dir())
        showinfo.assert_called_once_with("诊断目录", str(directory))
```

```python
def test_open_diagnostic_directory_uses_windows_startfile(self):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "diagnostics"
        app = object.__new__(AssistantApp)
        app.config_dir = directory
        with patch.object(
            app_module.os, "startfile", create=True
        ) as startfile:
            app.open_diagnostic_directory()
        startfile.assert_called_once_with(str(directory))
```

Run:

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest \
  test_assistant.LoopTests.test_open_diagnostic_directory_creates_and_shows_path_without_startfile \
  test_assistant.LoopTests.test_open_diagnostic_directory_uses_windows_startfile -v
```

Expected: missing-method errors.

- [ ] **Step 2: Implement native directory opening**

```python
def open_diagnostic_directory(self):
    self.config_dir.mkdir(parents=True, exist_ok=True)
    startfile = getattr(os, "startfile", None)
    if startfile is None:
        messagebox.showinfo("诊断目录", str(self.config_dir))
        return
    try:
        startfile(str(self.config_dir))
    except OSError as error:
        messagebox.showerror(
            "无法打开诊断目录", f"{error}\n{self.config_dir}"
        )
```

- [ ] **Step 3: Run directory tests to verify GREEN**

Run the Step 1 command.

Expected: both tests `OK`.

- [ ] **Step 4: Implement read-only preview UI**

Add `show_diagnostics()`:

```python
def show_diagnostics(self):
    window = tk.Toplevel(self.root)
    window.title("诊断截图")
    window.geometry("900x500")
    window.attributes("-topmost", True)
    window._diagnostic_images = []
    items = load_diagnostic_images(self.config_dir)

    if not any(item["path"].exists() for item in items):
        ttk.Label(
            window,
            text=(
                "尚无诊断截图\n"
                "连续三次无法把棋盘变化解释为合法走法后，程序会保存最近一组画面。\n"
                f"保存目录：{self.config_dir}"
            ),
            anchor="center",
            justify="center",
        ).pack(fill="both", expand=True, padx=20, pady=20)
    else:
        body = ttk.Frame(window, padding=12)
        body.pack(fill="both", expand=True)
        for column, item in enumerate(items):
            panel = ttk.Frame(body)
            panel.grid(row=0, column=column, sticky="nsew", padx=6)
            ttk.Label(panel, text=item["title"]).pack()
            if item["image"] is not None:
                photo = ImageTk.PhotoImage(item["image"])
                window._diagnostic_images.append(photo)
                ttk.Label(panel, image=photo).pack(pady=8)
            else:
                ttk.Label(panel, text=item["error"]).pack(pady=40)
            ttk.Label(
                panel,
                text=f"{item['path'].name}\n{item['modified']}\n{item['path']}",
                wraplength=400,
                justify="center",
            ).pack()
            body.columnconfigure(column, weight=1)

    ttk.Button(
        window,
        text="打开诊断目录",
        command=self.open_diagnostic_directory,
    ).pack(pady=(0, 12))
```

The path-existence condition ensures a missing pair shows the empty explanation, while one existing or corrupt file renders both panels and preserves each loader error.

- [ ] **Step 5: Add main-window button**

Increase geometry to `360x330`. Replace the single full-width sync button with:

```python
ttk.Button(
    frame, text="同步当前棋盘", command=self.sync_current_board
).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))
ttk.Button(
    frame, text="查看诊断", command=self.show_diagnostics
).grid(row=4, column=2, sticky="ew", padx=(8, 0), pady=(0, 4))
```

- [ ] **Step 6: Run all loop tests**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.LoopTests -v
```

Expected: all `LoopTests` pass.

---

### Task 4: Documentation and complete verification

**Files:**
- Modify: `README.md:80-105`
- Verify: `app.py`, `board_recognition.py`, `test_assistant.py`

**Interfaces:**
- No new runtime interfaces.

- [ ] **Step 1: Update user guidance**

Document:

- “查看诊断” opens the latest reference/current pair inside the app.
- “打开诊断目录” opens `%APPDATA%\XiangqiAssistant` on Windows.
- If no image exists, the preview explains the trigger condition and actual path.
- A newly rebuilt EXE is required; the old “当前行棋” build does not contain this feature.

- [ ] **Step 2: Run the complete suite**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant -v
```

Expected: all available tests pass; only the existing absent Tencent screenshot test may skip.

- [ ] **Step 3: Compile changed modules**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m py_compile app.py board_recognition.py game_state.py pikafish_engine.py test_assistant.py
```

Expected: exit code `0`, no output.

- [ ] **Step 4: Run real engine smoke test**

```bash
cd /Users/tang/PycharmProjects/pythonProject/xiangqi
python3 -m unittest test_assistant.EngineTests.test_real_engine_returns_legal_uci_move -v
```

Expected: `OK`.

- [ ] **Step 5: Record Windows boundary**

The macOS environment cannot validate `os.startfile`, Windows Tk rendering, JJ capture, or the rebuilt EXE. After `build_exe.bat`, manually trigger recovery and verify preview, timestamps, full paths, partial-image errors and Explorer opening. Do not claim Windows validation before it is run.
