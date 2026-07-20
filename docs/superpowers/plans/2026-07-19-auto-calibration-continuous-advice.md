# 自动校准与连续提示修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 标准开局自动定位棋盘，自动失败时允许手动拖框，并确保每次对手合法落子后继续给出下一步建议。

**Architecture:** 保留现有采集、识别、轮次跟踪、Pikafish 和 Tkinter 主链。`board_recognition.py` 只负责从标准开局截图推断 9×10 网格矩形并复用 `Calibration.create`；`app.py` 先自动校准，失败才调用已有拖框。连续提示修复放在识别层，避免单个低相似格点否决整盘，同时仍用棋子数量、双方将帅和合法着法约束阻止错误局面进入状态机。

**Tech Stack:** Python 3、OpenCV、NumPy、Tkinter、Pillow、unittest；不增加依赖。

## Global Constraints

- Windows 10，支持雷电模拟器和腾讯手游助手独立的“JJ象棋”窗口。
- 置顶窗口仅显示中文着法，不语音、不联网、不自动落子。
- 自动校准优先，手动拖框兜底；标准开局才允许建立棋子模板。
- 修改必须有失败测试、全量测试和语法检查。

---

### Task 1: 自动定位标准棋盘

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Consumes: OpenCV BGR 截图。
- Produces: `detect_board_rect(image) -> tuple[int, int, int, int]`；棋盘方向不影响几何定位。

- [x] 用可重复生成的标准开局截图写自动定位测试；用户附件当前已不在工作区，真实腾讯截图测试保留为“样本存在时运行”。
- [x] 运行目标测试，确认因接口不存在而失败。
- [x] 用标准开局 18 个底线棋子的圆形/行列规律实现最小定位算法，不复制校准逻辑。
- [x] 运行目标测试并确认通过。

### Task 2: 自动优先、手动兜底交互

**Files:**
- Modify: `app.py`
- Test: `test_assistant.py`

**Interfaces:**
- Consumes: `detect_board_rect` 与已有 `choose_board_rect`。
- Produces: `create_calibration(frame, rotated, source_id, choose_manual_rect)`；返回校准对象和校准方式。

- [x] 写自动成功不打开拖框、自动失败才打开拖框的测试。
- [x] 运行目标测试并确认失败。
- [x] 修改校准入口，自动成功显示“自动校准成功”，失败时明确提示并进入手动拖框。
- [x] 运行目标测试并确认通过。

### Task 3: 落子后识别与连续提示

**Files:**
- Modify: `board_recognition.py`
- Test: `test_assistant.py`

**Interfaces:**
- Consumes: 90 个格点分类分数。
- Produces: 合法的 `Recognition`；局面仍交给已有 `TurnTracker.observe` 判断轮次。

- [x] 写一个“少量变化格点低分但整体明确、红黑各走一步后再次提示”的回归测试。
- [x] 运行目标测试，确认旧的全盘最小值策略失败。
- [x] 将置信度聚合改为抗单点异常的保守分位数，并保留棋子上限、将帅数量和合法着法校验。
- [x] 运行目标测试和已有轮次测试并确认通过。

### Task 4: 操作文档与验证

**Files:**
- Modify: `README.md`

- [x] 写清采集源选择、自动校准、何时出现手动拖框、连续提示状态和常见错误。
- [x] 全量单进程超过执行通道约 30 秒上限；按测试类拆分运行 33 项，32 项通过，1 项因真实腾讯截图附件缺失跳过，0 失败、0 错误。
- [x] 运行 `python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py`，退出码 0。
- [x] 项目不是 Git 仓库，已逐项复查本次涉及的 `app.py`、`board_recognition.py`、`test_assistant.py`、`README.md` 和计划文件。
