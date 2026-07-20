# 象棋跨帧走法恢复与引擎窗口隐藏 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一次截图跨过红黑两步时仍顺序推进局面，并在 Windows 隐藏 Pikafish 控制台。

**Architecture:** 复用现有变化格评分和合法走法验证，最多枚举两个连续走法；`MotionTracker` 用待提交队列保持 `TurnTracker` 的单步接口。Pikafish 仅增加 Windows 原生无窗口创建标志。

**Tech Stack:** Python 3、OpenCV、NumPy、unittest、Windows subprocess；不增加依赖。

## Global Constraints

- 不改变现有 Tkinter 事件协议、Pikafish UCI 协议和公开 `Recognition` 结构。
- 不确定时等待，不输出未经合法性验证的建议。
- 最多恢复两步，不实现任意长度历史搜索。
- 项目目录不是 Git 仓库，无法执行提交步骤。

---

### Task 1: 跨帧两步恢复

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Consumes: 已知棋盘、最多 6 个变化交点、当前行棋方。
- Produces: `_infer_move_sequence_from_changes(...) -> tuple[tuple, ...] | None`；`MotionTracker.observe(...)` 仍返回 `Recognition`。

- [x] **Step 1: 写失败测试**

构造标准开局参考帧和同时包含红马、黑马两步的最终帧，断言 `MotionTracker` 连续返回红方中间局面与两步后的最终局面。

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.RecognitionTests.test_motion_tracker_recovers_two_moves_from_one_frame`

Expected: FAIL，第二个返回局面仍只有红方一步。

- [x] **Step 3: 写最小实现**

枚举当前方一步及对方紧接的一步，按已解释变化点数和分数选择唯一序列；`MotionTracker` 将第二个局面放入待提交队列。

- [x] **Step 4: 运行目标测试和识别测试**

Run: `python3 -m unittest -v test_assistant.RecognitionTests`

Expected: PASS，真实截图附件测试允许 SKIP。

### Task 2: 隐藏 Pikafish 控制台

**Files:**
- Modify: `pikafish_engine.py`
- Test: `test_assistant.py`

**Interfaces:**
- Produces: Windows 的 `Popen` 参数包含 `creationflags=subprocess.CREATE_NO_WINDOW`；其他平台为 0。

- [x] **Step 1: 写失败测试**

模拟 Windows 和 Pikafish 启动握手，检查 `subprocess.Popen` 收到无窗口标志。

- [x] **Step 2: 运行失败测试**

Run: `python3 -m unittest -v test_assistant.EngineTests.test_windows_engine_process_has_no_console`

Expected: FAIL，旧参数没有 `creationflags`。

- [x] **Step 3: 写最小实现**

在现有 `Popen` 调用增加平台条件的 `creationflags`，不改变管道参数。

- [x] **Step 4: 运行引擎测试**

Run: `python3 -m unittest -v test_assistant.EngineTests`

Expected: PASS。

### Task 3: 完整验证

**Files:**
- Test: `test_assistant.py`

- [x] **Step 1: 运行完整测试**

Run: `python3 -m unittest -v test_assistant.py`

Expected: 0 failures、0 errors；缺少 `123.jpg` 时 1 skip。

- [x] **Step 2: 运行语法检查**

Run: `python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py test_assistant.py`

Expected: exit code 0。
