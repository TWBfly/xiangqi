# 高亮棋子识别安全修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 防止可信圆占用证据对应的高亮棋子被模板分类静默识别为空，从而向 Pikafish 发送缺子局面并显示不可执行着法。

**Architecture:** 在现有 `Calibration.recognize()` 内分离占用与身份：可信圆命中的格只允许非空身份竞争；库存和位置修正后检查占用不变量，无法恢复时返回无效识别。保留 `best_advice()` 的真实棋规检查作为第二道防线，不修改 Pikafish。

**Tech Stack:** Python 3.10+、OpenCV 4.8.1.78、NumPy 1.26.1、标准库 `unittest`、本地 Pikafish UCI。

## Global Constraints

- 所有生产改动只写入外层仓库 `dev-1.0` 分支。
- `/xiangqi/pikafish` 是独立仓库，只读，不修改。
- 无法可靠确认棋子身份时必须拒绝分析，不得猜测后调用引擎。
- 不新增依赖、不引入多帧投票或新的识别状态机。
- 保留未跟踪用户文件 `3.png`，不加入提交。

### Task 1: 建立高亮棋子漏识别的失败测试

**Files:**
- Modify: `/Users/tang/PycharmProjects/pythonProject/xiangqi/test_assistant.py`（`RecognitionTests`）
- Test: 同上

**Interfaces:**
- Consumes: 现有 `standard_board_image()`、`Calibration.create()`、`grid_points()`。
- Produces: 一个可重复证明“占用格不能被空标签覆盖”的单元测试。

- [ ] **Step 1: Write the failing test**

在 `RecognitionTests` 中加入：

```python
def test_trusted_occupancy_keeps_piece_when_empty_template_wins(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    points = grid_points(rect)
    row, col = 0, 2  # 标准开局黑象，可信圆占用格
    piece_vector = calibration._vector(
        image, points[row][col], calibration.size
    )
    calibration.samples[calibration.labels == ""] = piece_vector
    calibration.samples[calibration.labels == "b"] *= 0.8

    result = calibration.recognize(image)

    self.assertTrue(result.valid, result.error)
    self.assertEqual(result.board[row][col], "b")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_trusted_occupancy_keeps_piece_when_empty_template_wins
```

Expected: FAIL，当前实现把 `(0, 2)` 识别为空，因为空模板分数高于黑象模板。

### Task 2: 让可信占用证据优先于空模板

**Files:**
- Modify: `/Users/tang/PycharmProjects/pythonProject/xiangqi/board_recognition.py:413-618`
- Test: `/Users/tang/PycharmProjects/pythonProject/xiangqi/test_assistant.py`

**Interfaces:**
- Consumes: Task 1 的失败测试；现有 `cell_scores`、`occupied`、`occupancy_trusted`、库存上限逻辑。
- Produces: `Calibration.recognize(image) -> Recognition`；可信圆命中的格不再静默变空。

- [ ] **Step 1: Write the minimal implementation**

在计算 `occupied` 和 `occupancy_trusted` 后，重建 `assigned` 时使用以下规则：可信占用格从非空标签中取最高分；未占用格保留现有强模板兜底：

```python
assigned = []
for cell, values in enumerate(cell_scores):
    row, col = divmod(cell, 9)
    if occupancy_trusted and (row, col) in occupied:
        nonempty = {label: score for label, score in values.items() if label}
        label = max(nonempty, key=nonempty.get) if nonempty else ""
    else:
        label = max(values, key=values.get)
    assigned.append(label)
```

保留现有库存调整，但在库存调整、仕相清理之后加入占用不变量；若不满足则返回无效识别：

```python
if occupancy_trusted:
    missing = sorted(
        occupied
        - {
            divmod(cell, 9)
            for cell, label in enumerate(assigned)
            if label
        }
    )
    if missing:
        return Recognition(
            board,
            confidence,
            False,
            f"检测到棋子但无法确认身份: {missing}",
        )
```

把已有测试 `test_circle_board_uses_structure_when_absolute_scores_are_low` 保留为结构识别兼容性测试；Task 1 只要求占用格不能因空模板丢失，不改变未占用格的强模板兜底。

- [ ] **Step 2: Run focused tests**

Run:

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_trusted_occupancy_keeps_piece_when_empty_template_wins \
  test_assistant.RecognitionTests.test_full_recognition_ignores_small_cursor_circle_on_empty_point \
  test_assistant.RecognitionTests.test_piece_inventory_resolves_moved_cannon_visual_distractor \
  test_assistant.RecognitionTests.test_circle_board_still_rejects_missing_king
```

Expected: all PASS。

- [ ] **Step 3: Commit**

```bash
git add board_recognition.py test_assistant.py
git commit -m "fix: preserve trusted occupied pieces"
```

### Task 3: 固化现场炮路安全回归

**Files:**
- Modify: `/Users/tang/PycharmProjects/pythonProject/xiangqi/test_assistant.py`（`EngineTests` 或 `LoopTests`）

**Interfaces:**
- Consumes: `AssistantApp.best_advice()`、`detect_move()`、固定现场 FEN。
- Produces: 现场真实盘面不会接受 `h7c7`，漏掉 `e7` 的错误盘面可被测试明确区分。

- [ ] **Step 1: Add the characterization test**

加入固定盘面测试。该测试覆盖已有的第二道防线，不修改生产代码；它必须在任何识别器修复前保持通过：

```python
def test_best_advice_rejects_cannon_jump_over_two_screens(self):
    fen = (
        "1rbaka2r/9/c1n1b1cCn/p1p1p3p/9/2P3p2/"
        "P3P3P/2N1C4/4N4/R1BAKAB1R w - - 0 1"
    )
    app = object.__new__(AssistantApp)
    app.stop_event = threading.Event()
    app.closing = False
    app.engine_lock = threading.Lock()
    app.engine = None
    app.best_move = Mock(return_value=("h7c7", "+3.88"))
    board, _ = br.fen_to_board(fen)

    with self.assertRaisesRegex(RuntimeError, "非法着法"):
        app.best_advice(board, f"position fen {fen}", "w", 1000)
```

- [ ] **Step 2: Run the characterization test**

Run:

```bash
python3 -m unittest -v test_assistant.LoopTests.test_best_advice_rejects_cannon_jump_over_two_screens
```

Expected: PASS；若失败，先停止执行并修复测试夹具或重新审查 `best_advice()`，不得绕过该防线继续修改识别器。

- [ ] **Step 3: Keep implementation unchanged**

不为已存在的合法性防线增加重复代码。只在 Task 2 的识别结果安全不变量上修复。

- [ ] **Step 4: Run focused regression**

```bash
python3 -m unittest -v \
  test_assistant.LoopTests.test_best_advice_rejects_cannon_jump_over_two_screens \
  test_assistant.EngineTests.test_best_advice_restarts_after_illegal_elephant_move
```

Expected: all PASS。

### Task 4: 全量验证与 Windows 交付检查

**Files:**
- Modify: 无新增生产文件；如测试文案需要同步，只修改 `test_assistant.py`。

**Interfaces:**
- Consumes: Tasks 1–3 的提交。
- Produces: 可复核的测试、编译、格式和真实引擎证据。

- [ ] **Step 1: Run the complete test suite**

```bash
python3 -m unittest discover -v
```

Expected: 0 failures；腾讯截图样本缺失导致的既有 skip 可以保留。

- [ ] **Step 2: Run static checks**

```bash
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
git diff --check
```

Expected: 两条命令退出码均为 0。

- [ ] **Step 3: Run real-engine position comparison**

用真实盘面 FEN 和删除 `e7` 的错误 FEN 分别调用本地 Pikafish；记录真实盘面不得返回 `h7c7`，错误 FEN 可以返回 `h7c7`，证明引擎行为与识别层根因一致。

- [ ] **Step 4: Check repository boundaries**

```bash
git status --short --branch
git -C pikafish status --short --branch
```

Expected：外层仅保留用户未跟踪 `3.png`（若仍存在）和本次已提交内容；嵌套 `pikafish` 仓库无修改。

- [ ] **Step 5: Commit any documentation-only test wording changes**

```bash
git add test_assistant.py
git commit -m "test: cover cannon path recognition safety"
```

仅在 Task 3 实际新增了测试时执行；若已有测试覆盖现场防线，不创建空提交。
