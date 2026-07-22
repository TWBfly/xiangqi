# Horse/Pawn Recognition Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent midgame grid drift and ambiguous horse/pawn template scores from producing a wrong FEN or an impossible “前兵进一” recommendation.

**Architecture:** Keep the existing OpenCV calibration classifier. Replace raw inlier-count alignment with a robust clipped-distance selector, trigger local resampling on identity ambiguity, and reject any final occupied label whose best-vs-runner-up margin is below `0.08`.

**Tech Stack:** Python 3, NumPy 1.26.1, OpenCV 4.8.1.78, `unittest`, existing Pikafish UCI wrapper.

## Global Constraints

- Target Windows 10 while keeping macOS regression execution working.
- Do not add Tesseract, EasyOCR, PaddleOCR, a CNN model, or any dependency.
- Do not modify the nested `pikafish` repository or UCI protocol.
- Keep `MIN_IDENTITY_SCORE = 0.35` behavior and add `MIN_IDENTITY_MARGIN = 0.08`.
- Do not lower global recognition confidence or coordinate legality checks.
- Do not change `_vector()` or `CALIBRATION_FEATURE_VERSION` unless a moved-horse characterization test proves the current feature fails after alignment is fixed.
- `app.py`, `board_recognition.py`, `pikafish_engine.py`, and `test_assistant.py` already contain user changes. Apply narrow patches and do not stage or commit these files as a whole.
- Approved design: `docs/superpowers/specs/2026-07-22-horse-pawn-recognition-safety-design.md`.

---

### Task 1: Select midgame grid translation by robust error

**Files:**
- Modify: `board_recognition.py:36-50,714-774`
- Test: `test_assistant.py:161-1094`

**Interfaces:**
- Consumes: `grid_points(rect, slant)`, detected circle centers shaped `(N, 2)`, calibration `rect`, `slant`, and `size`.
- Produces: `_best_grid_shift(centers, rect, slant, size) -> tuple[int, int] | None`.

- [ ] **Step 1: Add the field-circle failing test**

Add to `RecognitionTests`:

```python
def test_midgame_alignment_ignores_false_circle_match_count(self):
    centers = np.array(
        [
            (321, 266), (155, 269), (380, 269), (44, 274),
            (213, 274), (271, 277), (155, 382), (382, 382),
            (189, 407), (495, 433), (269, 437), (43, 439),
            (157, 439), (227, 463), (323, 507), (259, 509),
            (203, 511), (154, 545), (268, 547), (35, 597),
            (496, 597), (355, 625), (266, 649), (93, 705),
            (152, 765), (325, 765), (382, 767), (211, 770),
            (268, 771),
        ],
        dtype=np.float32,
    )
    selector = getattr(br, "_best_grid_shift", lambda *_args: (-2, 6))
    self.assertEqual(selector(centers, (42, 272, 494, 766), 0.0, 47), (0, 0))
```

- [ ] **Step 2: Run the test and observe RED**

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_midgame_alignment_ignores_false_circle_match_count
```

Expected: FAIL with `(-2, 6) != (0, 0)`, reproducing the current count-first result.

- [ ] **Step 3: Implement the pure robust selector**

Add after `grid_points()`:

```python
def _best_grid_shift(centers, rect, slant, size):
    left, top, right, bottom = rect
    tolerance_sq = (size * 0.32) ** 2
    best_score = None
    best_shift = None
    for dx in range(-40, 41, 2):
        for dy in range(-40, 41, 2):
            shifted = (left + dx, top + dy, right + dx, bottom + dy)
            points = np.asarray(grid_points(shifted, slant)).reshape(-1, 2)
            distances_sq = np.min(
                np.sum((centers[:, None, :] - points[None, :, :]) ** 2, axis=2),
                axis=1,
            )
            matches = int(np.count_nonzero(distances_sq <= tolerance_sq))
            if matches < 8:
                continue
            score = (
                -float(np.mean(np.minimum(distances_sq, tolerance_sq))),
                matches,
                -(abs(dx) + abs(dy)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_shift = (dx, dy)
    return best_shift
```

- [ ] **Step 4: Run the selector test and observe GREEN**

Run the Step 2 command.

Expected: PASS with `(0, 0)`.

- [ ] **Step 5: Replace the inline count-first loop**

In the Hough fallback, replace the nested candidate loop with:

```python
shift = _best_grid_shift(centers, self.rect, self.slant, self.size)
if shift is not None:
    dx, dy = shift
    aligned_rect = (left + dx, top + dy, right + dx, bottom + dy)
```

- [ ] **Step 6: Run existing alignment regressions**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_midgame_alignment_ignores_false_circle_match_count \
  test_assistant.RecognitionTests.test_alignment_recovers_same_size_negative_client_area_shift \
  test_assistant.RecognitionTests.test_alignment_falls_back_when_starting_rect_is_unavailable
```

Expected: all three tests PASS.

- [ ] **Step 7: Inspect the narrow diff without committing dirty files**

```bash
git diff -- board_recognition.py test_assistant.py
```

Expected: the new alignment hunks only add the selector, replace the inline objective, and add its regression.

---

### Task 2: Measure and enforce per-cell identity margin

**Files:**
- Modify: `board_recognition.py:30-34,776-888,989-1049`
- Test: `test_assistant.py:161-1094`

**Interfaces:**
- Consumes: per-cell score dictionaries shaped `{label: float}` and the final assigned label.
- Produces: `_identity_margin(scores, label) -> float`, ambiguity-triggered neighborhood search, and invalid `Recognition` with candidate details.

- [ ] **Step 1: Add the identity-margin failing test**

Add to `RecognitionTests`:

```python
def test_identity_margin_compares_final_label_with_runner_up(self):
    margin = getattr(br, "_identity_margin", lambda _scores, _label: 1.0)
    self.assertAlmostEqual(
        margin({"": 0.90, "N": 0.74, "P": 0.71, "R": 0.20}, "N"),
        0.03,
    )
```

- [ ] **Step 2: Run the margin test and observe RED**

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_identity_margin_compares_final_label_with_runner_up
```

Expected: FAIL because the fallback returns `1.0` instead of `0.03`.

- [ ] **Step 3: Add constants and the minimum helper**

Near the existing calibration constants add:

```python
MIN_IDENTITY_SCORE = 0.35
MIN_IDENTITY_MARGIN = 0.08
```

After `recognition_confidence()` add:

```python
def _identity_margin(scores, label):
    runner_up = max(
        (score for candidate, score in scores.items() if candidate and candidate != label),
        default=-1.0,
    )
    return scores[label] - runner_up
```

- [ ] **Step 4: Run the margin test and observe GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Add a failing ambiguous horse/pawn integration test**

Add to `RecognitionTests`:

```python
def test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    points = grid_points(rect)
    size, half = calibration.size, calibration.size // 2
    current = image.copy()

    def crop(row, col):
        x, y = points[row][col]
        return image[y - half : y - half + size, x - half : x - half + size].copy()

    def paste(row, col, value):
        x, y = points[row][col]
        current[y - half : y - half + size, x - half : x - half + size] = value

    empty = crop(5, 5)
    paste(9, 1, empty)
    paste(6, 0, empty)
    paste(3, 2, crop(9, 1))

    horse = calibration.samples[np.flatnonzero(calibration.labels == "N")[0]]
    pawn = calibration.samples[np.flatnonzero(calibration.labels == "P")[0]]
    ambiguous = horse * 0.49 + pawn * 0.51
    ambiguous /= np.linalg.norm(ambiguous)
    original_vector = calibration._vector
    target_x, target_y = points[3][2]

    def mocked_vector(frame, point, sample_size):
        if abs(point[0] - target_x) <= 3 and abs(point[1] - target_y) <= 3:
            return ambiguous.copy()
        return original_vector(frame, point, sample_size)

    with patch.object(calibration, "_vector", side_effect=mocked_vector):
        result = calibration.recognize(current, align_grid=False)

    self.assertFalse(result.valid)
    self.assertRegex(result.error, r"棋子身份不明确.*\(3, 2\).*N=.*P=")
```

- [ ] **Step 6: Run the integration test and tune only its synthetic vector until RED is causal**

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn
```

Expected before production behavior changes: FAIL because recognition accepts a structurally valid board rather than returning `棋子身份不明确`. If the synthetic mixture produces a different non-identity failure, adjust only the `0.49/0.51` weights until the current code accepts `P` at `(3, 2)`; do not change production code during RED setup.

- [ ] **Step 7: Trigger local search on a small identity margin**

After initial `scores` are built, determine the highest nonempty label and use:

```python
piece_labels = [label for label in scores if label]
best_piece = max(piece_labels, key=scores.get, default="")
identity_ambiguous = bool(
    best_piece and _identity_margin(scores, best_piece) < MIN_IDENTITY_MARGIN
)
if not is_empty and (scores[best_label] < 0.70 or identity_ambiguous):
```

Keep the existing ±3 search and early exit behavior unchanged.

- [ ] **Step 8: Add the final identity gate after board/confidence construction**

Before returning a valid recognition, inspect every final nonempty assignment:

```python
for cell, label in enumerate(assigned):
    if not label:
        continue
    label_score = cell_scores[cell][label]
    margin = _identity_margin(cell_scores[cell], label)
    if label_score >= MIN_IDENTITY_SCORE and margin >= MIN_IDENTITY_MARGIN:
        continue
    screen_row, screen_col = divmod(cell, 9)
    board_position = (
        (9 - screen_row, 8 - screen_col)
        if self.rotated
        else (screen_row, screen_col)
    )
    candidates = sorted(
        (
            (score, candidate)
            for candidate, score in cell_scores[cell].items()
            if candidate
        ),
        reverse=True,
    )[:2]
    detail = " / ".join(
        f"{candidate}={score:.2f}" for score, candidate in candidates
    )
    return Recognition(
        board,
        confidence,
        False,
        f"棋子身份不明确: {board_position}，候选 {detail}",
    )
```

Replace the existing literal `0.35` occupied-cell threshold with `MIN_IDENTITY_SCORE` so the rule has one source.

- [ ] **Step 9: Run identity tests and observe GREEN**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_identity_margin_compares_final_label_with_runner_up \
  test_assistant.RecognitionTests.test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn
```

Expected: both tests PASS.

- [ ] **Step 10: Run inventory and occupancy safety regressions**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_piece_inventory_resolves_moved_cannon_visual_distractor \
  test_assistant.RecognitionTests.test_trusted_occupancy_rejects_piece_when_identity_is_unclear \
  test_assistant.RecognitionTests.test_occupied_cell_never_demoted_to_empty \
  test_assistant.RecognitionTests.test_full_recognition_limits_low_confidence_duplicate_piece
```

Expected: all existing tests PASS. If an existing synthetic fixture is intentionally ambiguous, update only that fixture to provide an identity margin of at least `0.08`; do not weaken the production guard.

---

### Task 3: Characterize moved horses and lock the c6c7 rule boundary

**Files:**
- Modify: `test_assistant.py:760-820,1600-1630`
- Modify: `app.py:38`

**Interfaces:**
- Consumes: current `_vector()`, `Calibration.recognize()`, and `is_legal_xiangqi_move()`.
- Produces: permanent moved-horse and exact field-coordinate regressions; visible version `2026.07.22.3`.

- [ ] **Step 1: Add moved red/black horse characterization tests**

Add to `RecognitionTests`:

```python
def test_calibration_keeps_moved_horses_distinct_from_pawns(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    points = grid_points(rect)
    size, half = calibration.size, calibration.size // 2

    for source, target, expected in (
        ((9, 1), (3, 2), "N"),
        ((0, 1), (5, 2), "n"),
    ):
        with self.subTest(source=source, target=target):
            current = image.copy()

            def crop(row, col):
                x, y = points[row][col]
                return image[
                    y - half : y - half + size,
                    x - half : x - half + size,
                ].copy()

            def paste(row, col, value):
                x, y = points[row][col]
                current[
                    y - half : y - half + size,
                    x - half : x - half + size,
                ] = value

            paste(source[0], source[1], crop(5, 5))
            paste(target[0], target[1], crop(source[0], source[1]))
            result = calibration.recognize(current, align_grid=False)
            self.assertTrue(result.valid, result.error)
            self.assertEqual(result.board[target[0]][target[1]], expected)
```

- [ ] **Step 2: Run the characterization test**

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_calibration_keeps_moved_horses_distinct_from_pawns
```

Expected: PASS. If it fails specifically because `_vector()` cannot distinguish the moved horse after Tasks 1–2, stop and return to design review before changing the feature format.

- [ ] **Step 3: Add the exact c6c7 rule regression**

Extend `test_is_legal_xiangqi_move_horse_and_flying_general`:

```python
field = [list(row) for row in STANDARD_BOARD]
field[9][1] = ""
field[3][2] = "N"
field = tuple(tuple(row) for row in field)
self.assertFalse(is_legal_xiangqi_move(field, 3, 2, 2, 2, "w"))
self.assertTrue(is_legal_xiangqi_move(field, 3, 2, 1, 1, "w"))
```

- [ ] **Step 4: Run the exact rule test**

```bash
python3 -m unittest -v test_assistant.LoopTests.test_is_legal_xiangqi_move_horse_and_flying_general
```

Expected: PASS, proving the rule layer was already correct once identity is `N`.

- [ ] **Step 5: Update the visible version**

Change:

```python
APP_VERSION = "2026.07.22.3"
```

- [ ] **Step 6: Run the focused recognition suite**

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_midgame_alignment_ignores_false_circle_match_count \
  test_assistant.RecognitionTests.test_identity_margin_compares_final_label_with_runner_up \
  test_assistant.RecognitionTests.test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn \
  test_assistant.RecognitionTests.test_calibration_keeps_moved_horses_distinct_from_pawns \
  test_assistant.LoopTests.test_is_legal_xiangqi_move_horse_and_flying_general
```

Expected: all five tests PASS.

---

### Task 4: Full verification and field handoff

**Files:**
- Verify: `app.py`
- Verify: `board_recognition.py`
- Verify: `pikafish_engine.py`
- Verify: `game_state.py`
- Verify: `test_assistant.py`

**Interfaces:**
- Consumes: robust alignment and identity safety from Tasks 1–3.
- Produces: fresh evidence that the field drift is corrected, ambiguous identity cannot reach FEN, and legal engine paths remain intact.

- [ ] **Step 1: Run the complete suite**

```bash
python3 -m unittest test_assistant.py
```

Expected: all tests PASS; the optional Tencent screenshot test may remain skipped when its fixture is absent.

- [ ] **Step 2: Compile runtime modules**

```bash
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Replay the field circle centers**

Run the Task 1 field-circle test again.

Expected: `_best_grid_shift(...) == (0, 0)`, replacing the old `(-2, 6)` drift.

- [ ] **Step 4: Run real Pikafish regressions**

```bash
python3 -m unittest -v \
  test_assistant.EngineTests.test_real_engine_survives_rejected_invalid_position \
  test_assistant.EngineTests.test_real_engine_returns_legal_uci_move \
  test_assistant.EngineTests.test_real_engine_analyzes_reported_midgame_position
```

Expected: all three tests PASS.

- [ ] **Step 5: Verify dependencies and nested repository are untouched**

```bash
git diff -- requirements.txt
git -C pikafish status --short
```

Expected: both commands produce no output.

- [ ] **Step 6: Check patch hygiene and working-tree ownership**

```bash
git diff --check
git status --short
```

Expected: report the known pre-existing trailing whitespace separately; no new warning may originate from added lines. Keep the four pre-dirty implementation files unstaged.

- [ ] **Step 7: Review acceptance requirements**

Confirm from fresh output:

```text
[ ] field circles select (0, 0), not (-2, 6)
[ ] moved red and black horses remain N/n at pawn-legal coordinates
[ ] ambiguous N/P identity returns invalid Recognition with coordinate and candidates
[ ] invalid Recognition does not generate FEN or call Pikafish
[ ] c6c7 is rejected when c6 contains N
[ ] legal UCI analysis remains operational
[ ] no OCR dependency or nested Pikafish change
```

- [ ] **Step 8: Leave dirty implementation files uncommitted**

Do not stage `app.py`, `board_recognition.py`, `pikafish_engine.py`, or `test_assistant.py`. Report the design/plan commits separately so the user can later consolidate their existing working-tree changes intentionally.
