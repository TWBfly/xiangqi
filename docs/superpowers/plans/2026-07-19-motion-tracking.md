# Motion-Based Continuous Advice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用前后截图的交叉点变化推导合法着法，消除落子后棋子身份误分类导致的卡死。

**Architecture:** 校准阶段保留一次完整标准局面识别；对局阶段新增一个小型状态对象，保存已接受棋盘和参考帧，并用现有走法合法性函数筛选运动候选。`AssistantApp.run_loop` 只消费该状态对象输出的局面，原有Pikafish、队列和UI不改。

**Tech Stack:** Python 3、OpenCV、NumPy、unittest；不增加依赖。

## Global Constraints

- Windows 10，支持雷电模拟器和腾讯手游助手JJ象棋窗口。
- 自动校准优先、手动拖框兜底。
- 不联网、不语音、不自动落子。
- 不确定时等待，不输出未经合法性验证的建议。

---

### Task 1: 运动着法推导

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Consumes: `Calibration`、当前已知棋盘、参考帧、当前帧、行棋方。
- Produces: `infer_move_from_frames(...) -> tuple | None`。

- [x] 写红炮移动、黑马移动、无变化、普通捕获和同字异色捕获的失败测试。
- [x] 运行目标测试，确认接口缺失而失败；同字异色捕获随后也先复现为失败。
- [x] 复用 `_piece_can_move`、`_king_in_check` 和90点颜色特征，实现最多6个变化格的合法候选评分。
- [x] 运行目标测试并确认通过。

### Task 2: 连续状态跟踪

**Files:**
- Modify: `board_recognition.py`
- Modify: `app.py`
- Test: `test_assistant.py`

**Interfaces:**
- Produces: `MotionTracker.observe(frame) -> Recognition`，只有两帧一致候选才更新局面。

- [x] 写“初始建议→红走→黑走→第二次建议”、两帧稳定、第三噪声格和高亮消失测试。
- [x] 运行目标测试，确认旧全盘分类产生“马”超限且运动接口缺失。
- [x] 在 `run_loop` 中用 `MotionTracker` 替代每帧全盘棋子分类。
- [x] 运行目标测试并确认通过。

### Task 3: 状态文案和验证

**Files:**
- Modify: `README.md`
- Test: `test_assistant.py`

- [x] 将内部编码错误改成中文棋子名，并说明“等待棋盘稳定/重新校准”的含义。
- [x] 分组运行41项测试：40项通过、1项因真实腾讯截图附件缺失跳过，0失败、0错误。
- [x] 运行Python语法检查，退出码0。
- [x] 独立审查误报、漏报和Windows实机边界；Critical/Important均已关闭。
