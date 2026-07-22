# Pikafish Position Validation Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure impossible Xiangqi piece coordinates are rejected locally before any `position fen` command can terminate Pikafish.

**Architecture:** Keep `is_coordinate_legal()` as the single coordinate rule source, reuse it from recognition and `validate_board()`, and validate at both FEN serialization and the external-process boundary. Deterministic input errors propagate without engine restart; process EOF waits briefly for the real exit code.

**Tech Stack:** Python 3 standard library, NumPy/OpenCV already present in the project, `unittest`, Pikafish UCI.

## Global Constraints

- Target Windows 10 behavior while keeping macOS test execution working.
- Do not lower recognition confidence thresholds or change image-classification tuning.
- Do not add dependencies or modify the nested `pikafish` repository.
- Reuse `is_coordinate_legal()`, `validate_board()`, and `fen_to_board()`; do not create a new rules module.
- `app.py`, `board_recognition.py`, `pikafish_engine.py`, and `test_assistant.py` contain pre-existing user changes. Apply narrow patches and do not stage or commit these dirty files as a whole.
- The approved design is `docs/superpowers/specs/2026-07-22-pikafish-position-validation-design.md`.

---

### Task 1: Complete and centralize coordinate legality

**Files:**
- Modify: `board_recognition.py:52-71,372-435,990-1025`
- Test: `test_assistant.py:132-158,319-338`

**Interfaces:**
- Consumes: `is_coordinate_legal(label: str, r: int, c: int) -> bool` and `normalize_board(board, rotated=False) -> tuple`.
- Produces: complete pawn coordinate rules, `validate_board(board) -> str` using the shared rule, and `board_to_fen(board, active="w") -> str` that raises `ValueError` for an invalid board.

- [ ] **Step 1: Add failing pawn-coordinate tests**

Add a dedicated test after `test_coordinate_legal_filtering`:

```python
def test_uncrossed_pawns_only_use_starting_files(self):
    from board_recognition import is_coordinate_legal

    self.assertTrue(is_coordinate_legal("p", 3, 0))
    self.assertFalse(is_coordinate_legal("p", 3, 1))
    self.assertFalse(is_coordinate_legal("p", 4, 1))
    self.assertTrue(is_coordinate_legal("p", 5, 1))
    self.assertTrue(is_coordinate_legal("P", 6, 0))
    self.assertFalse(is_coordinate_legal("P", 6, 1))
    self.assertFalse(is_coordinate_legal("P", 5, 1))
    self.assertTrue(is_coordinate_legal("P", 4, 1))
```

- [ ] **Step 2: Run the coordinate test and observe RED**

Run:

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_uncrossed_pawns_only_use_starting_files
```

Expected: FAIL because `is_coordinate_legal("p", 3, 1)` currently returns `True`.

- [ ] **Step 3: Implement the minimum complete pawn rule**

Replace the two partial pawn branches in `is_coordinate_legal()`:

```python
if label == "p" and (r < 3 or (r < 5 and c % 2)):
    return False
if label == "P" and (r > 6 or (r > 4 and c % 2)):
    return False
```

Replace the post-classification duplicated advisor/bishop coordinate sets with the existing helper:

```python
mutable_board = [list(row) for row in board]
corrected = False
for r in range(10):
    for c in range(9):
        if mutable_board[r][c] and not is_coordinate_legal(
            mutable_board[r][c], r, c
        ):
            mutable_board[r][c] = ""
            corrected = True
if corrected:
    board = tuple(tuple(row) for row in mutable_board)
```

- [ ] **Step 4: Run the coordinate test and observe GREEN**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Add failing board/FEN boundary tests**

Add these tests to the existing board-recognition test class:

```python
def test_validate_board_rejects_uncrossed_black_pawn_on_odd_file(self):
    illegal = moved(STANDARD_BOARD, (3, 0), (3, 1))
    self.assertEqual(br.validate_board(illegal), "黑卒在非法坐标: (3, 1)")

def test_fen_parser_rejects_reported_invalid_black_pawn(self):
    fen = (
        "2bakab1r/9/7c1/pp4p1p/2p6/9/P1P1P1P1P/"
        "2N1C4/9/2BAKABNR w - - 0 1"
    )
    with self.assertRaisesRegex(ValueError, r"黑卒.*\(3, 1\)"):
        br.fen_to_board(fen)

def test_fen_writer_rejects_uncrossed_black_pawn_on_odd_file(self):
    illegal = moved(STANDARD_BOARD, (3, 0), (3, 1))
    with self.assertRaisesRegex(ValueError, r"黑卒.*\(3, 1\)"):
        board_to_fen(illegal, "w")
```

- [ ] **Step 6: Run the three boundary tests and observe RED**

Run:

```bash
python3 -m unittest -v \
  test_assistant.BoardTests.test_validate_board_rejects_uncrossed_black_pawn_on_odd_file \
  test_assistant.BoardTests.test_fen_parser_rejects_reported_invalid_black_pawn \
  test_assistant.BoardTests.test_fen_writer_rejects_uncrossed_black_pawn_on_odd_file
```

Expected: failures because total validation omits pawns and the FEN writer does not validate.

- [ ] **Step 7: Centralize total validation and protect serialization**

In `validate_board()`, replace the duplicated advisor/bishop coordinate sets with:

```python
coordinate_names = {
    "k": "黑将",
    "K": "红帅",
    "a": "黑士",
    "A": "红仕",
    "b": "黑象",
    "B": "红相",
    "p": "黑卒",
    "P": "红兵",
}
for r, row in enumerate(board):
    for c, piece in enumerate(row):
        if piece and not is_coordinate_legal(piece, r, c):
            return f"{coordinate_names[piece]}在非法坐标: {(r, c)}"
```

In `board_to_fen()`, immediately after checking `active`, add:

```python
error = validate_board(board)
if error:
    raise ValueError(error)
```

- [ ] **Step 8: Run all Task 1 tests and observe GREEN**

Run both commands from Steps 2 and 6.

Expected: all four tests PASS.

- [ ] **Step 9: Verify the narrow diff without committing dirty files**

Run:

```bash
git diff -- board_recognition.py test_assistant.py
```

Expected: the new hunks are limited to pawn legality, shared validation, FEN writer validation, and their tests; pre-existing hunks remain untouched.

---

### Task 2: Reject invalid FEN before the process boundary

**Files:**
- Modify: `pikafish_engine.py:1-8,200-219`
- Modify: `app.py:729-750`
- Test: `test_assistant.py:1155-1375`

**Interfaces:**
- Consumes: `fen_to_board(fen: str) -> tuple[tuple, str]`.
- Produces: `PikafishEngine.get_best_move()` that validates pure FEN and `position fen ... moves ...` before sending commands, and `AssistantApp.best_move()` that does not retry `ValueError`.

- [ ] **Step 1: Add failing process-boundary tests**

Add to `EngineTests`:

```python
def test_invalid_fen_is_rejected_before_any_engine_command(self):
    fen = (
        "2bakab1r/9/7c1/pp4p1p/2p6/9/P1P1P1P1P/"
        "2N1C4/9/2BAKABNR w - - 0 1"
    )
    for position in (fen, f"position fen {fen} moves a0a1"):
        with self.subTest(position=position):
            engine = object.__new__(PikafishEngine)
            engine._lock = threading.Lock()
            sent = []
            engine.send_command = sent.append
            engine._read_until = lambda _predicate, _timeout: ["bestmove a0a1"]
            with self.assertRaisesRegex(ValueError, r"黑卒.*\(3, 1\)"):
                engine.get_best_move(position, movetime=200)
            self.assertEqual(sent, [])

def test_app_does_not_retry_deterministic_position_error(self):
    app = object.__new__(AssistantApp)
    app.engine_lock = threading.Lock()
    app.stop_event = Mock(is_set=Mock(return_value=False))
    app.closing = False
    app.engine = Mock()
    app.engine.get_best_move.side_effect = ValueError("非法局面")

    with (
        patch("app.PikafishEngine") as engine_class,
        self.assertRaisesRegex(ValueError, "非法局面"),
    ):
        app.best_move("position fen invalid", 1000)

    app.engine.get_best_move.assert_called_once_with(
        "position fen invalid", movetime=1000
    )
    app.engine.close.assert_not_called()
    engine_class.assert_not_called()

def test_real_engine_survives_rejected_invalid_position(self):
    fen = (
        "2bakab1r/9/7c1/pp4p1p/2p6/9/P1P1P1P1P/"
        "2N1C4/9/2BAKABNR w - - 0 1"
    )
    engine = PikafishEngine()
    try:
        with self.assertRaisesRegex(ValueError, r"黑卒.*\(3, 1\)"):
            engine.get_best_move(fen, movetime=200)
        self.assertTrue(engine.is_alive())
    finally:
        engine.close()
```

- [ ] **Step 2: Run the three tests and observe RED**

Run:

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_invalid_fen_is_rejected_before_any_engine_command \
  test_assistant.EngineTests.test_app_does_not_retry_deterministic_position_error \
  test_assistant.EngineTests.test_real_engine_survives_rejected_invalid_position
```

Expected: the command-boundary test sends commands instead of raising, the app test calls/closes the engine twice, and the real-engine test fails because Pikafish exits with code 1.

- [ ] **Step 3: Add the minimal engine preflight**

Import the existing parser:

```python
from board_recognition import fen_to_board
```

In `get_best_move()`, after building `position_command` and before acquiring `_lock`, add:

```python
fen = position_command.removeprefix("position fen ").split(" moves ", 1)[0]
try:
    fen_to_board(fen)
except ValueError as error:
    raise ValueError(f"拒绝发送非法局面给Pikafish: {error}") from error
```

In `AssistantApp.best_move()`, add a dedicated branch before the generic exception cleanup:

```python
except ValueError:
    raise
```

- [ ] **Step 4: Run the three tests and observe GREEN**

Run the command from Step 2.

Expected: all three tests PASS, the invalid position sends no UCI commands, and the real process remains alive until the test closes it.

- [ ] **Step 5: Verify valid commands still pass**

Run:

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_engine_uses_adaptive_time_with_hard_limit \
  test_assistant.EngineTests.test_best_move_retries_engine_start_after_errno_22
```

Expected: both existing tests PASS.

---

### Task 3: Preserve the engine and make exit diagnostics deterministic

**Files:**
- Modify: `pikafish_engine.py:112-139`
- Modify: `app.py:38`
- Test: `test_assistant.py:1170-1195,1360-1375`

**Interfaces:**
- Consumes: `subprocess.Popen.wait(timeout: float) -> int` and `PikafishEngine.is_alive() -> bool`.
- Produces: EOF diagnostics containing a final numeric exit code when available and version `2026.07.22.2`.

- [ ] **Step 1: Add the failing exit-code race test**

Add to `EngineTests`:

```python
def test_read_until_waits_for_exit_code_after_stdout_eof(self):
    engine = object.__new__(PikafishEngine)
    engine._lines = Queue()
    engine.process = Mock()
    engine.process.poll.return_value = None
    engine.process.wait.return_value = 1
    engine._lines.put("info string CRITICAL ERROR")
    engine._lines.put(None)

    with self.assertRaisesRegex(RuntimeError, r"退出码 1 / 0x00000001"):
        engine._read_until(lambda line: False, 1.0)

    engine.process.wait.assert_called_once_with(timeout=0.2)
```

- [ ] **Step 2: Run the race test and observe RED**

Run:

```bash
python3 -m unittest -v test_assistant.EngineTests.test_read_until_waits_for_exit_code_after_stdout_eof
```

Expected: FAIL because the message currently contains `退出码 未知` and `wait()` is not called.

- [ ] **Step 3: Wait briefly for the final exit status**

Inside `_read_until()` when `line is None`, after the first `poll()` add:

```python
if code is None and self.process:
    try:
        code = self.process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
```

- [ ] **Step 4: Run diagnostic tests and observe GREEN**

Run:

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_read_until_waits_for_exit_code_after_stdout_eof \
  test_assistant.EngineTests.test_read_until_preserves_engine_exit_diagnostics
```

Expected: both tests PASS.

- [ ] **Step 5: Re-run the real-process survival regression**

Run the integration test first observed RED in Task 2:

```bash
python3 -m unittest -v test_assistant.EngineTests.test_real_engine_survives_rejected_invalid_position
```

Expected: PASS and the engine remains alive until the test closes it.

- [ ] **Step 6: Update the visible application version**

Change:

```python
APP_VERSION = "2026.07.22.2"
```

- [ ] **Step 7: Verify the focused implementation diff**

Run:

```bash
git diff -- app.py board_recognition.py pikafish_engine.py test_assistant.py
```

Expected: only approved position-validation, retry, diagnostics, version, and test hunks are newly added; unrelated pre-existing changes are preserved.

---

### Task 4: Full verification and handoff

**Files:**
- Verify: `app.py`
- Verify: `board_recognition.py`
- Verify: `pikafish_engine.py`
- Verify: `game_state.py`
- Verify: `test_assistant.py`

**Interfaces:**
- Consumes: all behavior produced by Tasks 1–3.
- Produces: fresh evidence that the user-reported FEN is blocked locally and legal analysis paths remain operational.

- [ ] **Step 1: Run the complete unit suite**

```bash
python3 -m unittest -v test_assistant.py
```

Expected: all tests PASS; the optional Tencent sample may remain skipped when its fixture is unavailable.

- [ ] **Step 2: Compile every touched runtime module**

```bash
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Reproduce the original FEN at both local boundaries**

```bash
python3 -c "from board_recognition import fen_to_board; fen_to_board('2bakab1r/9/7c1/pp4p1p/2p6/9/P1P1P1P1P/2N1C4/9/2BAKABNR w - - 0 1')"
```

Expected: `ValueError: 黑卒在非法坐标: (3, 1)`.

Run the real-engine survival test from Task 3 again.

Expected: PASS and the engine stays alive until the test closes it.

- [ ] **Step 4: Run legal real-engine regressions**

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_real_engine_returns_legal_uci_move \
  test_assistant.EngineTests.test_real_engine_analyzes_reported_midgame_position
```

Expected: both tests PASS with a four-character UCI move.

- [ ] **Step 5: Check patch hygiene**

```bash
git diff --check
git status --short
```

Expected: report any pre-existing trailing whitespace separately; no new whitespace error may come from the added hunks. The four source/test files remain modified because they were already dirty and are intentionally not committed wholesale.

- [ ] **Step 6: Review requirements line by line**

Confirm from test output and diff:

```text
[ ] illegal pawn candidate filtered during recognition
[ ] validate_board and both FEN boundaries reject (3, 1)
[ ] invalid input sends no UCI command
[ ] deterministic ValueError is not retried
[ ] real Pikafish remains alive after rejection
[ ] exit code race reports numeric status
[ ] legal FEN and position-with-moves remain compatible
[ ] no dependency or nested-engine change
```

- [ ] **Step 7: Leave implementation uncommitted and report exact files**

Do not stage the four pre-dirty implementation files. Report the design/plan document commits separately from the working-tree implementation so the user can decide how to consolidate their existing changes.
