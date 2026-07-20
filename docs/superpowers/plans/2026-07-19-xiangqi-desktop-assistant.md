# Windows 象棋桌面辅助器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Flask 网页程序改造成 Windows 10 置顶象棋辅助器，校准后持续读取雷电模拟器或腾讯手游助手，只在用户回合显示中文推荐着法。

**Architecture:** 使用 ADB 截图并以 Windows 窗口截图兜底；校准把当前皮肤和棋盘矩形转换为 90 个固定交点模板；纯逻辑回合跟踪器只接受稳定、可解释的单步局面变化；Tkinter 主线程显示状态，后台线程执行截图、识别和 Pikafish 分析。

**Tech Stack:** Python 3.10+、标准库 `tkinter/ctypes/subprocess/threading/unittest`、现有 OpenCV/NumPy/Pillow、现有 Pikafish。

## Global Constraints

- 目标平台是 Windows 10，兼容雷电模拟器和腾讯手游助手。
- 界面只使用置顶小窗口，不打开网页，不播放语音，不自动落子。
- 首次使用及皮肤、方向、分辨率变化后允许在标准开局重新校准。
- 不增加第三方依赖；删除 Flask 依赖。
- 识别不可信时不得向引擎提交局面。
- 当前目录不是 Git 仓库，计划中的每项任务以测试通过作为检查点，不执行提交命令。

---

## 文件结构

- Create: `capture.py` — ADB 发现、PNG 截图、Windows 顶层窗口发现和窗口截图。
- Rewrite: `board_recognition.py` — 标准开局、校准数据、90 点识别、FEN、走子差异和回合跟踪。
- Modify: `pikafish_engine.py` — 有超时的 UCI 通信、引擎就绪检查和完整中文记谱。
- Rewrite: `app.py` — Tkinter 置顶窗口、校准画布和后台辅助循环。
- Create: `test_assistant.py` — 使用标准库 `unittest` 的最小回归测试。
- Modify: `build_exe.bat` — 无控制台 GUI 打包并移除网页资源。
- Modify: `requirements.txt` — 移除 Flask。
- Rewrite: `README.md` — Windows 安装、校准和使用说明。
- Delete: `templates/index.html`、`piece_templates/*.png` — 删除网页与失效的固定皮肤模板。

### Task 1: 采集源发现和 PNG 解码

**Files:**
- Create: `capture.py`
- Create: `test_assistant.py`

**Interfaces:**
- Produces: `parse_adb_devices(output: str) -> list[str]`
- Produces: `find_adb() -> str | None`
- Produces: `adb_frame(adb: str, serial: str, timeout: float = 3.0) -> numpy.ndarray`
- Produces: `list_emulator_windows() -> list[tuple[int, str]]`
- Produces: `window_frame(hwnd: int) -> numpy.ndarray`

- [ ] **Step 1: 写 ADB 设备解析失败测试**

```python
class CaptureTests(unittest.TestCase):
    def test_parse_adb_devices_keeps_only_online_devices(self):
        output = "List of devices attached\nemulator-5554\tdevice\n127.0.0.1:5555\toffline\nABC\tunauthorized\n"
        self.assertEqual(parse_adb_devices(output), ["emulator-5554"])
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.CaptureTests`

Expected: FAIL，因为 `capture` 模块尚不存在。

- [ ] **Step 3: 实现最小采集模块**

```python
def parse_adb_devices(output):
    return [parts[0] for line in output.splitlines()
            if len(parts := line.split()) == 2 and parts[1] == "device"]

def adb_frame(adb, serial, timeout=3.0):
    data = subprocess.run(
        [adb, "-s", serial, "exec-out", "screencap", "-p"],
        check=True, capture_output=True, timeout=timeout,
    ).stdout
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("模拟器截图不是有效PNG")
    return frame
```

`find_adb()` 只检查 `PATH`、已保存路径和雷电/腾讯常见目录；`list_emulator_windows()` 与 `window_frame()` 在非 Windows 平台返回空列表或抛出明确错误，在 Windows 使用 `ctypes.windll.user32` 和 Pillow `ImageGrab.grab(bbox=...)`。

- [ ] **Step 4: 验证 GREEN**

Run: `python3 -m unittest -v test_assistant.CaptureTests`

Expected: PASS，1 test。

### Task 2: 棋盘网格、标准开局和 FEN

**Files:**
- Rewrite: `board_recognition.py`
- Modify: `test_assistant.py`

**Interfaces:**
- Produces: `STANDARD_BOARD: tuple[tuple[str, ...], ...]`
- Produces: `grid_points(rect: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]`
- Produces: `normalize_board(board, rotated: bool) -> tuple[tuple[str, ...], ...]`
- Produces: `board_to_fen(board, active: str) -> str`

- [ ] **Step 1: 写网格、旋转和 FEN 测试**

```python
class BoardTests(unittest.TestCase):
    def test_standard_board_generates_initial_fen(self):
        self.assertEqual(
            board_to_fen(STANDARD_BOARD, "w"),
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )

    def test_grid_has_ninety_evenly_spaced_points(self):
        points = grid_points((36, 225, 511, 744))
        self.assertEqual((len(points), len(points[0])), (10, 9))
        self.assertEqual(points[0][0], (36, 225))
        self.assertEqual(points[-1][-1], (511, 744))

    def test_rotated_board_normalizes_to_standard(self):
        rotated = tuple(tuple(reversed(row)) for row in reversed(STANDARD_BOARD))
        self.assertEqual(normalize_board(rotated, True), STANDARD_BOARD)
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.BoardTests`

Expected: FAIL，因为新接口尚不存在。

- [ ] **Step 3: 实现纯棋盘函数**

使用不可变 10×9 元组表示棋盘。`grid_points()` 使用 `numpy.linspace` 并四舍五入；`normalize_board()` 在 `rotated=True` 时旋转 180 度；`board_to_fen()` 压缩连续空格并严格验证 `active in {"w", "b"}`。

- [ ] **Step 4: 验证 GREEN**

Run: `python3 -m unittest -v test_assistant.BoardTests`

Expected: PASS，3 tests。

### Task 3: 用 image.png 校准并逐格识别

**Files:**
- Modify: `board_recognition.py`
- Modify: `test_assistant.py`

**Interfaces:**
- Produces: `Calibration.create(image, rect, rotated=False) -> Calibration`
- Produces: `Calibration.recognize(image) -> Recognition`
- Produces: `Calibration.save(directory: Path) -> None`
- Produces: `Calibration.load(directory: Path) -> Calibration`
- Produces: `Recognition(board, confidence, valid, error)`

- [ ] **Step 1: 写真实截图回归测试**

```python
class RecognitionTests(unittest.TestCase):
    def test_calibration_recognizes_supplied_initial_board(self):
        image = cv2.imread(str(ROOT / "image.png"))
        calibration = Calibration.create(image, (36, 225, 511, 744))
        result = calibration.recognize(image)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)
        self.assertEqual(sum(bool(piece) for row in result.board for piece in row), 32)

    def test_calibration_round_trip_preserves_recognition(self):
        image = cv2.imread(str(ROOT / "image.png"))
        with tempfile.TemporaryDirectory() as tmp:
            Calibration.create(image, (36, 225, 511, 744)).save(Path(tmp))
            result = Calibration.load(Path(tmp)).recognize(image)
        self.assertEqual(result.board, STANDARD_BOARD)
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.RecognitionTests`

Expected: FAIL，因为 `Calibration` 尚不存在。

- [ ] **Step 3: 实现校准和识别**

校准从标准开局每个已知棋子位置提取多个 52×52 样本，并从空位提取空格样本。所有样本缩放到 40×40，使用中心圆形遮罩和灰度归一化。识别计算当前交点与每类样本的最大 `cv2.TM_CCOEFF_NORMED` 分数；空位和棋子共同参与分类。校准创建后必须用同一帧回读出 `STANDARD_BOARD`，否则抛出 `ValueError("标准开局校准失败")`。

配置元数据写入 `config.json`，样本数组写入 `samples.npz`。保存使用临时文件后 `Path.replace()`，避免留下半成品。

- [ ] **Step 4: 验证 GREEN**

Run: `python3 -m unittest -v test_assistant.RecognitionTests`

Expected: PASS，2 tests，识别 32 枚棋子。

### Task 4: 走子差异和用户回合状态机

**Files:**
- Modify: `board_recognition.py`
- Modify: `test_assistant.py`

**Interfaces:**
- Produces: `Move(side: str, start: tuple[int, int], end: tuple[int, int])`
- Produces: `detect_move(before, after) -> Move | None`
- Produces: `TurnTracker(player: str).observe(board) -> bool`
- `observe()` 返回 `True` 仅表示当前稳定局面首次进入用户回合、需要调用引擎。

- [ ] **Step 1: 写普通走子、吃子和回合测试**

```python
class TurnTests(unittest.TestCase):
    def test_detect_move_handles_quiet_move_and_capture(self):
        before = [list(row) for row in STANDARD_BOARD]
        after = [row[:] for row in before]
        after[7][1], after[7][4] = "", "C"
        self.assertEqual(detect_move(before, after), Move("w", (7, 1), (7, 4)))

    def test_red_gets_one_initial_analysis_then_waits_for_black(self):
        tracker = TurnTracker("w")
        self.assertTrue(tracker.observe(STANDARD_BOARD))
        self.assertFalse(tracker.observe(STANDARD_BOARD))
        after_red = moved(STANDARD_BOARD, (7, 1), (7, 4))
        self.assertFalse(tracker.observe(after_red))
        after_black = moved(after_red, (0, 1), (2, 2))
        self.assertTrue(tracker.observe(after_black))

    def test_unexplained_change_does_not_switch_turn(self):
        tracker = TurnTracker("w")
        tracker.observe(STANDARD_BOARD)
        broken = tuple(tuple("" for _ in range(9)) for _ in range(10))
        self.assertFalse(tracker.observe(broken))
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.TurnTests`

Expected: FAIL，因为走子检测和回合跟踪尚不存在。

- [ ] **Step 3: 实现最小状态机**

`detect_move()` 只接受两个变化格：起点由棋子变空，终点由空或异色棋子变为同一己方棋子。`TurnTracker` 从红方先行开始，缓存最后局面和已分析 FEN；无法解释的变化不更新最后可信局面。

- [ ] **Step 4: 验证 GREEN**

Run: `python3 -m unittest -v test_assistant.TurnTests`

Expected: PASS，3 tests。

### Task 5: 强化 Pikafish 和中文记谱

**Files:**
- Modify: `pikafish_engine.py`
- Modify: `test_assistant.py`

**Interfaces:**
- Preserves: `PikafishEngine.get_best_move(fen, movetime=1000) -> tuple[str, str]`
- Produces: `PikafishEngine.get_chinese_move(move, board) -> str`
- All startup and search reads have finite timeouts.

- [ ] **Step 1: 写中文记谱和真实引擎测试**

```python
class EngineTests(unittest.TestCase):
    def test_chinese_move_uses_board_before_move(self):
        self.assertEqual(PikafishEngine.get_chinese_move("h2e2", STANDARD_BOARD), "炮二平五")

    def test_real_engine_returns_legal_uci_move(self):
        engine = PikafishEngine()
        try:
            move, score = engine.get_best_move(board_to_fen(STANDARD_BOARD, "w"), movetime=200)
        finally:
            engine.close()
        self.assertRegex(move, r"^[a-i][0-9][a-i][0-9]$")
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.EngineTests`

Expected: 中文记谱测试暴露当前输入类型或同路棋子问题；超时接口尚未实现。

- [ ] **Step 3: 实现有界 UCI 通信**

用单个后台读线程把 stdout 行写入 `queue.Queue`，`_read_until(predicate, timeout)` 使用 `queue.get(timeout=...)`。启动顺序固定为 `uci -> uciok -> setoption EvalFile -> isready -> readyok`。搜索只发送 `go movetime N`。超时抛出 `TimeoutError`，不永久阻塞。

中文记谱接受不可变棋盘，先查起点棋子，再处理红黑列号、进退平和同路同类棋子的前后名称。

- [ ] **Step 4: 验证 GREEN**

Run: `python3 -m unittest -v test_assistant.EngineTests`

Expected: PASS，2 tests。

### Task 6: Tkinter置顶窗口和连续后台循环

**Files:**
- Rewrite: `app.py`
- Modify: `test_assistant.py`

**Interfaces:**
- Produces: `AssistantApp(root: tkinter.Tk)`
- Produces: `stable_board(previous, current, count) -> tuple[board | None, int]`
- Uses: capture source、`Calibration`、`TurnTracker`、`PikafishEngine`。

- [ ] **Step 1: 写稳定帧过滤测试**

```python
class LoopTests(unittest.TestCase):
    def test_board_requires_two_identical_frames(self):
        board, count = stable_board(None, STANDARD_BOARD, 0)
        self.assertIsNone(board)
        board, count = stable_board(STANDARD_BOARD, STANDARD_BOARD, count)
        self.assertEqual(board, STANDARD_BOARD)
```

- [ ] **Step 2: 验证 RED**

Run: `python3 -m unittest -v test_assistant.LoopTests`

Expected: FAIL，因为 `stable_board` 尚不存在。

- [ ] **Step 3: 实现最小 GUI 和循环**

窗口包含采集源、执棋方、校准、开始/停止、推荐着法和状态。调用 `root.attributes("-topmost", True)`。后台 daemon 线程每 700ms 采集；识别和引擎结果通过 `queue.Queue` 传回主线程，主线程用 `root.after(100, poll_results)` 更新控件。

校准窗口使用 Tkinter `Canvas` 显示当前截图，鼠标按下和释放得到棋盘矩形。校准成功后写入 `%APPDATA%\XiangqiAssistant`。关闭窗口先设置停止事件，再关闭引擎。

- [ ] **Step 4: 验证 GREEN 和导入安全**

Run: `python3 -m unittest -v test_assistant.LoopTests && python3 -c "import app; print('import ok')"`

Expected: PASS，且导入 `app` 不创建窗口、不启动线程。

### Task 7: 打包、清理和文档

**Files:**
- Modify: `build_exe.bat`
- Modify: `requirements.txt`
- Rewrite: `README.md`
- Delete: `templates/index.html`
- Delete: `piece_templates/*.png`

- [ ] **Step 1: 修改依赖和打包命令**

`requirements.txt` 只保留 OpenCV、NumPy、Pillow、PyInstaller。`build_exe.bat` 使用：

```bat
pyinstaller --clean --onefile --windowed ^
  --add-data "pikafish.nnue;." ^
  --add-data "pikafish.exe;." ^
  app.py -n "XiangqiAssistant"
```

删除模板网页和固定棋子模板，不打包 Flask 或网页资源。

- [ ] **Step 2: 重写README**

README 明确写出：标准开局校准、红黑方选择、ADB 优先/窗口兜底、窗口截图不可最小化、开始和停止辅助、重新校准条件、EXE 构建命令。

- [ ] **Step 3: 运行完整自动验证**

Run: `python3 -m unittest -v test_assistant.py`

Expected: 全部测试 PASS，0 failures，0 errors。

Run: `python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py`

Expected: exit 0，无输出。

- [ ] **Step 4: Windows人工构建与验收**

Run on Windows 10: `build_exe.bat`

Expected: `dist\XiangqiAssistant.exe` 生成；启动后只有 Tkinter 置顶窗口，不启动浏览器。连接雷电或腾讯手游助手，在标准开局完成校准后，红方首次显示推荐，用户和对手各走一步后自动显示下一条推荐。

## 计划自检结果

- 规格中的采集、校准、识别、旋转、回合、引擎、界面、打包和错误处理均有对应任务。
- 所有新增纯逻辑均先写失败测试再实现。
- Windows 专属窗口枚举和 `.bat` 无法在当前 macOS 工作区完成真实运行验证，最终状态必须明确报告为“代码与跨平台测试通过，Windows 实机验收待执行”，除非用户随后提供 Windows 执行结果。
- 未引入 Flask 替代品、视觉模型、自动落子或新依赖。
