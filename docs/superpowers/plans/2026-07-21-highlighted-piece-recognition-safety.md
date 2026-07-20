# Highlighted Piece Recognition Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a visually present highlighted piece from becoming an empty square and causing Pikafish to recommend a move that is impossible on the real board.

**Architecture:** Keep the existing OpenCV/template recognizer, but make trusted Hough occupancy authoritative over the empty template. Recover a non-empty identity only when its score reaches the existing calibration threshold, then fail closed if any detected occupied square is empty after inventory and position corrections. Keep the existing `best_advice()` legal-move check as the independent engine-output guard.

**Tech Stack:** Python 3.10+, OpenCV 4.8.1.78, NumPy 1.26.1, `unittest`, Pikafish UCI.

## Global Constraints

- Work only on branch `dev-1.0`.
- Do not modify `/Users/tang/PycharmProjects/pythonProject/xiangqi/pikafish`.
- Do not add dependencies, OCR, neural networks, databases, or a multi-frame voting framework.
- If a detected piece identity cannot be confirmed, reject analysis instead of displaying a guessed move.
- Preserve rotated-board behavior and the existing strong-template fallback for Hough misses.
- Preserve the user's untracked `3.png`.

---

### Task 1: Make trusted occupancy authoritative

**Files:**
- Modify: `test_assistant.py:676-739`
- Modify: `board_recognition.py:509-618`

**Interfaces:**
- Consumes: `Calibration.recognize(image) -> Recognition`, `Calibration._occupied_cells(image, rect, size) -> set[tuple[int, int]]`, existing `self.threshold`.
- Produces: the same `Recognition` interface; an invalid result uses the error prefix `检测到棋子但无法确认身份:`.

- [ ] **Step 1: Write the failing occupied-piece recovery test**

Add to `RecognitionTests`:

```python
    def test_circle_occupancy_keeps_piece_when_empty_template_wins(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        elephant = calibration.samples[calibration.labels == "b"][0].copy()
        calibration.samples[calibration.labels == ""] = elephant
        calibration.samples[calibration.labels == "b"] *= 0.8
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label
        }

        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            result = calibration.recognize(image)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)
```

- [ ] **Step 2: Write the failing uncertain-identity safety test**

```python
    def test_circle_occupancy_rejects_uncertain_piece_identity(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        elephant = calibration.samples[calibration.labels == "b"][0].copy()
        calibration.samples[calibration.labels == ""] = elephant
        calibration.samples[calibration.labels == "b"] *= 0.2
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label
        }

        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            result = calibration.recognize(image)

        self.assertFalse(result.valid)
        self.assertIn("检测到棋子但无法确认身份", result.error)
```

- [ ] **Step 3: Run both tests and verify RED**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_circle_occupancy_keeps_piece_when_empty_template_wins \
  test_assistant.RecognitionTests.test_circle_occupancy_rejects_uncertain_piece_identity
```

Expected: first test fails because both black elephants become empty; second fails because that recognition is incorrectly marked valid.

- [ ] **Step 4: Replace occupancy filtering with explicit recovery**

In `Calibration.recognize()`, replace the current occupancy list comprehension with:

```python
        uncertain = []
        if occupancy_trusted:
            filtered = []
            strong_score = max(
                0.55,
                1.0001
                if self.threshold + 0.10 >= 1.0
                else self.threshold + 0.10,
            )
            for cell, label in enumerate(assigned):
                position = divmod(cell, 9)
                if position in occupied and not label:
                    nonempty = [candidate for candidate in labels if candidate]
                    label = max(nonempty, key=cell_scores[cell].get)
                    if cell_scores[cell][label] < self.threshold:
                        uncertain.append(position)
                elif (
                    position not in occupied
                    and (not label or cell_scores[cell][label] < strong_score)
                ):
                    label = ""
                filtered.append(label)
            assigned = filtered
```

Do not add a classifier, state machine, or configuration option.

- [ ] **Step 5: Add the post-correction occupancy invariant**

After restricted-piece correction has produced the final `board`, add:

```python
        missing = (
            occupied
            - {
                (row, col)
                for row, values in enumerate(board)
                for col, label in enumerate(values)
                if label
            }
            if occupancy_trusted
            else set()
        )
        unresolved = sorted(set(uncertain) | missing)
```

Replace the current `error` choice with:

```python
        if unresolved:
            squares = ", ".join(
                f"{chr(97 + col)}{9 - row}" for row, col in unresolved
            )
            error = f"检测到棋子但无法确认身份: {squares}"
        else:
            error = (
                validate_board(board)
                if occupancy_trusted
                else self._validate(board, confidence)
            )
```

- [ ] **Step 6: Run focused recognition tests and verify GREEN**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_circle_occupancy_keeps_piece_when_empty_template_wins \
  test_assistant.RecognitionTests.test_circle_occupancy_rejects_uncertain_piece_identity \
  test_assistant.RecognitionTests.test_full_recognition_ignores_small_cursor_circle_on_empty_point \
  test_assistant.RecognitionTests.test_full_recognition_never_assigns_piece_without_large_circle \
  test_assistant.RecognitionTests.test_piece_inventory_resolves_moved_cannon_visual_distractor
```

Expected: all five tests pass.

- [ ] **Step 7: Commit the recognition fix**

```bash
git add board_recognition.py test_assistant.py
git commit -m "fix: preserve highlighted pieces during recognition"
```

---

### Task 2: Lock the reported cannon move behind the legal-move guard

**Files:**
- Modify: `test_assistant.py:1060-1090`

**Interfaces:**
- Consumes: `AssistantApp.best_advice(board, position, side, movetime)`, `board_to_fen()`, and the existing `detect_move()` guard.
- Produces: a regression proving `h7c7` cannot be displayed when both real blockers exist.

- [ ] **Step 1: Add the exact reported-position regression test**

```python
    def test_best_advice_rejects_reported_cannon_move_with_two_screens(self):
        board, _side = br.fen_to_board(
            "1rbaka2r/9/c1n1b1cCn/p1p1p3p/9/2P3p2/"
            "P3P3P/2N1C4/4N4/R1BAKAB1R w - - 0 1"
        )
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.engine = Mock()
        app.best_move = Mock(return_value=("h7c7", "+3.88"))

        with self.assertRaisesRegex(RuntimeError, "h7c7"):
            app.best_advice(
                board,
                f"position fen {board_to_fen(board, 'w')}",
                "w",
                1000,
            )

        self.assertEqual(app.best_move.call_count, 2)
```

- [ ] **Step 2: Run the exact regression**

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_best_advice_rejects_reported_cannon_move_with_two_screens
```

Expected: PASS without production changes. It records that the second guard works when recognition supplies the true board; Task 1 fixes the upstream data loss.

- [ ] **Step 3: Commit the现场 regression**

```bash
git add test_assistant.py
git commit -m "test: cover reported illegal cannon advice"
```

---

### Task 3: Verify the full pipeline

**Files:**
- Verify only: `board_recognition.py`, `app.py`, `pikafish_engine.py`, `game_state.py`, `test_assistant.py`
- Preserve: `3.png`

**Interfaces:**
- Consumes: all recognition and engine interfaces after Tasks 1–2.
- Produces: fresh evidence that the fix is syntactically valid, regression-safe, and does not mutate the nested engine repository.

- [ ] **Step 1: Run the full automated suite**

```bash
python3 -m unittest discover -v
```

Expected: all tests pass; the existing Tencent screenshot test may remain skipped when its sample is absent.

- [ ] **Step 2: Compile Python modules**

```bash
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Re-run the现场 engine contrast**

```bash
python3 -c 'from pikafish_engine import PikafishEngine; positions={"真实盘面":"1rbaka2r/9/c1n1b1cCn/p1p1p3p/9/2P3p2/P3P3P/2N1C4/4N4/R1BAKAB1R w - - 0 1","漏掉e7黑象":"1rbaka2r/9/c1n3cCn/p1p1p3p/9/2P3p2/P3P3P/2N1C4/4N4/R1BAKAB1R w - - 0 1"}; e=PikafishEngine(); [(print(name,e.get_best_move(fen,5000))) for name,fen in positions.items()]; e.close()'
```

Expected: the漏象局面 reproduces `h7c7` near `+3.88`; the true board does not return `h7c7`.

- [ ] **Step 4: Check repository hygiene**

```bash
git diff --check
git status --short --branch
git -C pikafish status --short --branch
```

Expected: outer repository contains only intentional commits plus preserved untracked `3.png`; nested Pikafish remains clean on `master`.
