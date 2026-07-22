# 校准真值与默认搜索时间 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止非标准开局生成校准模板，淘汰旧污染样本，并把默认搜索时间改为 20 秒。

**Architecture:** 在 `Calibration.create()` 的模板提取前复用 `_occupied_cells()` 做独立标准占用验证；升级校准版本；保持保存、识别和 Pikafish 接口不变。

**Tech Stack:** Python 3、OpenCV、NumPy、unittest、Pikafish UCI。

## Global Constraints

- 不新增 OCR、模型或依赖。
- 不修改嵌套 Pikafish。
- 先 RED 后生产代码。
- 不提交混有用户修改的实现文件。

---

### Task 1: 标准开局独立真值门

**Files:**
- Modify: `test_assistant.py`
- Modify: `board_recognition.py:528-620`

**Interfaces:**
- Consumes: `_occupied_cells(image, rect, size)`、旋转后的 `STANDARD_BOARD`。
- Produces: 非标准占用时 `Calibration.create()` 抛出 `ValueError`。

- [ ] 写测试：模拟标准占用集合少一个炮、多一个中盘落点，即使 `recognize()` 被模拟为标准棋盘，也必须拒绝。
- [ ] 运行测试，确认当前代码因没有抛错而 RED。
- [ ] 在提取样本前比较 `actual_occupied` 与 `expected_occupied`，错误包含缺失和多余坐标。
- [ ] 运行标准开局自动/手动校准和新测试，确认 GREEN。

---

### Task 2: 淘汰旧污染样本

**Files:**
- Modify: `board_recognition.py:32`
- Modify: `test_assistant.py`

**Interfaces:**
- Consumes: `config.json.feature_version`。
- Produces: 只加载版本 `2` 的样本。

- [ ] 将现有旧格式测试改为显式写入 `feature_version=1`，运行确认当前版本 1 错误接受。
- [ ] 把 `CALIBRATION_FEATURE_VERSION` 改为 `2`。
- [ ] 运行保存/加载、损坏样本、旧格式测试。

---

### Task 3: 默认 20 秒和 GUI 文案

**Files:**
- Modify: `app.py:36-38,190`
- Modify: `test_assistant.py`

**Interfaces:**
- Consumes: `DEFAULT_SEARCH_SECONDS`。
- Produces: GUI 初始搜索秒数 `20`，按钮“标准开局校准”。

- [ ] 把默认值测试期望从 10 改为 20，运行确认 RED。
- [ ] 设置 `DEFAULT_SEARCH_SECONDS = 20`，更新测试名、按钮文案和 `APP_VERSION = "2026.07.23.2"`。
- [ ] 运行默认值、时间解析和界面初始化相关测试。

---

### Task 4: 全量验收

**Files:**
- Verify: `app.py`
- Verify: `board_recognition.py`
- Verify: `pikafish_engine.py`
- Verify: `test_assistant.py`

- [ ] 运行校准、识别、将军应对和真实引擎聚焦测试。
- [ ] 运行 `python3 -m unittest test_assistant.py`。
- [ ] 运行 `python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py`。
- [ ] 检查 `requirements.txt`、嵌套 `pikafish`、`git diff --check` 和工作区所有权。
