# 将军应对与高亮棋子识别修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止高亮黑车被识别成黑卒，并拒绝任何走后仍被将军的建议着法。

**Architecture:** 复用 `Calibration.recognize()` 已有的 ±3 像素邻域模板搜索，对 Hough 确认的占用格无条件启用；复用已有 `_king_in_check()` 补全共享合法性函数。Pikafish 保持原样。

**Tech Stack:** Python 3、OpenCV、NumPy、unittest、UCI/Pikafish。

## Global Constraints

- 不新增 OCR、模型或第三方依赖。
- 不改变校准特征版本。
- 不修改 `/xiangqi/pikafish`。
- 生产代码必须先有因正确原因失败的测试。
- 保留用户现有未提交修改，不整体提交脏文件。

---

### Task 1: 锁定被将军后的完整合法性

**Files:**
- Modify: `test_assistant.py`
- Modify: `board_recognition.py:106-260`

**Interfaces:**
- Consumes: `is_legal_xiangqi_move(board, sr, sc, er, ec, player)`、`_king_in_check(board, side)`。
- Produces: 走后己方将帅仍被攻击时返回 `False`。

- [ ] **Step 1: 写现场局面失败测试**

```python
def test_move_is_illegal_when_it_does_not_evade_rook_check(self):
    board, side = br.fen_to_board(
        "1rb1k4/4a4/2Ra5/pNpC2p1p/4r4/8P/2P6/8B/9/2BAKA3 "
        "w - - 0 1"
    )
    self.assertTrue(br._king_in_check(board, side))
    self.assertFalse(is_legal_xiangqi_move(board, 3, 1, 5, 2, side))
```

- [ ] **Step 2: 运行并确认 RED**

```bash
python3 -m unittest -v test_assistant.LoopTests.test_move_is_illegal_when_it_does_not_evade_rook_check
```

预期：最后一个断言失败，当前函数错误返回 `True`。

- [ ] **Step 3: 最小修复共享合法性函数**

```python
temp_board = [list(row) for row in board]
temp_board[er][ec] = temp_board[sr][sc]
temp_board[sr][sc] = ""
side = "w" if piece_is_red else "b"
return not _king_in_check(tuple(tuple(row) for row in temp_board), side)
```

删除原来只检查将帅同列照面的重复代码。

- [ ] **Step 4: 运行 GREEN 与既有走法测试**

```bash
python3 -m unittest -v \
  test_assistant.LoopTests.test_move_is_illegal_when_it_does_not_evade_rook_check \
  test_assistant.LoopTests.test_is_legal_xiangqi_move_horse_and_flying_general \
  test_assistant.LoopTests.test_is_legal_xiangqi_move_cannon_rules
```

预期：全部通过。

---

### Task 2: 让高置信度错误类别也执行邻域纠偏

**Files:**
- Modify: `test_assistant.py`
- Modify: `board_recognition.py:850-915`

**Interfaces:**
- Consumes: `occupied`、`scores`、现有 `_vector()` 和 ±3 邻域搜索。
- Produces: Hough 确认占用的格点始终进行邻域模板比较。

- [ ] **Step 1: 写高置信度黑车/黑卒失败测试**

创建标准校准图，把黑车从 `(0,0)` 移到 `(4,4)`，清空 `(3,4)` 黑卒；占用集合与该局面一致。模拟目标中心返回 `0.95 * pawn_vector`，目标邻域一个点返回精确 `rook_vector`。断言结果有效且 `(4,4) == "r"`。

- [ ] **Step 2: 运行并确认 RED**

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_occupied_high_confidence_wrong_label_searches_neighborhood
```

预期：当前中心分数高、间隔大，不搜索邻域，目标被识别为 `p` 或最终断言失败。

- [ ] **Step 3: 扩展现有搜索条件**

```python
if not is_empty and (
    (r_idx, c_idx) in occupied
    or scores[best_label] < 0.70
    or identity_ambiguous
):
```

不创建新的搜索器或配置项。

- [ ] **Step 4: 运行 GREEN 与识别回归**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_occupied_high_confidence_wrong_label_searches_neighborhood \
  test_assistant.RecognitionTests.test_calibration_keeps_moved_horses_distinct_from_pawns \
  test_assistant.RecognitionTests.test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn \
  test_assistant.RecognitionTests.test_piece_inventory_recognizes_moved_cannon
```

预期：全部通过。

---

### Task 3: 端到端与真实引擎验收

**Files:**
- Modify: `app.py:38`
- Verify: `pikafish_engine.py`
- Verify: `test_assistant.py`

**Interfaces:**
- Consumes: 正确现场 FEN、真实 `PikafishEngine`、`AssistantApp.best_advice()`。
- Produces: 版本 `2026.07.23.1` 和完整验证证据。

- [ ] **Step 1: 增加真实引擎和输出边界测试**

正确现场 FEN 必须由真实 Pikafish 返回合法应将着；模拟 `best_move()` 返回 `b6c4` 时，`best_advice()` 必须拒绝。

- [ ] **Step 2: 更新版本号**

```python
APP_VERSION = "2026.07.23.1"
```

- [ ] **Step 3: 运行完整测试与编译**

```bash
python3 -m unittest test_assistant.py
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

预期：全部测试通过；可选现场夹具可跳过；编译退出码为 0。

- [ ] **Step 4: 检查依赖和仓库边界**

```bash
git diff --exit-code -- requirements.txt
git -C pikafish status --short
git diff --check
git status --short
```

预期：依赖和嵌套 Pikafish 无变化；仅报告工作区原有脏文件和已知尾随空格。
