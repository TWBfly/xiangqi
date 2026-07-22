# Stable Board Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct same-size Windows client-area board drift before feature extraction and reject calibration samples created by an incompatible feature schema.

**Architecture:** `Calibration.recognize()` will reuse the existing strong starting-position detector before its current Hough translation fallback. Calibration persistence will carry one explicit feature version so old vectors cannot silently enter the current classifier. Pikafish and capture code remain unchanged.

**Tech Stack:** Python 3, OpenCV, NumPy, stdlib `unittest`, existing Tkinter application.

## Global Constraints

- Target Windows 10, JJ 象棋, window capture and ADB capture.
- Do not lower `Calibration.threshold == 0.45`.
- Do not add dependencies or modify the nested `pikafish` repository.
- Preserve all pre-existing uncommitted changes in `app.py`, `board_recognition.py`, `pikafish_engine.py`, and `test_assistant.py`.
- Do not commit shared dirty implementation files; verify the incremental diff instead.

---

### Task 1: Reproduce and fix same-size board drift

**Files:**
- Modify: `test_assistant.py` in `RecognitionTests`
- Modify: `board_recognition.py` in `Calibration.recognize`

**Interfaces:**
- Consumes: `detect_board_rect(image) -> tuple[int, int, int, int]`
- Produces: `Calibration.recognize(image, align_grid=True) -> Recognition` using a validated per-frame rectangle

- [ ] **Step 1: Write the failing translation regression test**

Add this test beside the existing automatic calibration tests:

```python
def test_alignment_recovers_same_size_negative_client_area_shift(self):
    image, rect = standard_board_image()
    calibration = Calibration.create(image, rect)
    shifted = cv2.warpAffine(
        image,
        np.float32([[1, 0, -16], [0, 1, -16]]),
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    result = calibration.recognize(shifted, align_grid=True)

    self.assertTrue(result.valid, result.error)
    self.assertEqual(result.board, STANDARD_BOARD)
    self.assertGreaterEqual(result.confidence, calibration.threshold)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_alignment_recovers_same_size_negative_client_area_shift
```

Expected before the fix: `FAIL`; the current Hough-only alignment returns an invalid board for `(-16, -16)`.

- [ ] **Step 3: Implement strong rectangle detection before Hough fallback**

Replace only the opening of the current `align_grid` block in `Calibration.recognize()` so it follows this structure; retain the existing Hough body inside the `if aligned_rect == self.rect:` fallback:

```python
aligned_rect = self.rect
if align_grid:
    left, top, right, bottom = self.rect
    try:
        detected_rect = detect_board_rect(image)
    except ValueError:
        detected_rect = None

    if detected_rect:
        detected_width = detected_rect[2] - detected_rect[0]
        detected_height = detected_rect[3] - detected_rect[1]
        if (
            abs(detected_width - (right - left)) <= self.size * 0.5
            and abs(detected_height - (bottom - top)) <= self.size * 0.5
        ):
            aligned_rect = detected_rect

    if aligned_rect == self.rect:
        try:
            gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
            minimum_radius = max(8, int(self.size * 0.34))
            circles = _detect_circles_adaptive(
                gray,
                minDist=max(12, self.size * 0.6),
                minRadius=minimum_radius,
                maxRadius=max(minimum_radius + 2, int(self.size * 0.7)),
                param2_start=28,
                target_count=16,
            )
            if circles is not None:
                centers = circles[0][:, :2]
                valid_mask = (
                    (centers[:, 1] >= max(0, top - 60))
                    & (centers[:, 1] <= min(image.shape[0], bottom + 30))
                )
                centers = centers[valid_mask]
                best_dx, best_dy = 0, 0
                max_score = -999.0
                for dx in range(-40, 41, 2):
                    for dy in range(-40, 41, 2):
                        shifted = (
                            left + dx,
                            top + dy,
                            right + dx,
                            bottom + dy,
                        )
                        pts = np.array(
                            grid_points(shifted, self.slant)
                        ).reshape(-1, 2)
                        dists_sq = np.min(
                            np.sum(
                                (centers[:, None, :] - pts[None, :, :]) ** 2,
                                axis=2,
                            ),
                            axis=0,
                        )
                        matches = int(
                            np.count_nonzero(
                                dists_sq <= (self.size * 0.32) ** 2
                            )
                        )
                        score = matches - 0.08 * (abs(dx) + abs(dy))
                        if score > max_score:
                            max_score = score
                            best_dx, best_dy = dx, dy
                if max_score >= 8.0:
                    aligned_rect = (
                        left + best_dx,
                        top + best_dy,
                        right + best_dx,
                        bottom + best_dy,
                    )
        except Exception:
            pass
```

Do not persist `detected_rect` into `self.rect`; alignment is frame-local. Do not accept implicit scaling beyond half a calibrated cell in either board dimension.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command again.

Expected: `OK`; restored board equals `STANDARD_BOARD` and confidence is at least `0.45`.

- [ ] **Step 5: Verify both alignment paths**

Run:

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_alignment_recovers_same_size_negative_client_area_shift \
  test_assistant.RecognitionTests.test_calibration_recognizes_piece_moved_to_new_intersection \
  test_assistant.RecognitionTests.test_rotated_calibration_normalizes_screen_orientation
```

Expected: all three tests pass. This task is not committed because both modified files contained user changes before this plan.

---

### Task 2: Reject incompatible persisted calibration features

**Files:**
- Modify: `test_assistant.py` imports and `RecognitionTests`
- Modify: `board_recognition.py` constants, `Calibration.save`, and `Calibration.load`

**Interfaces:**
- Produces: `CALIBRATION_FEATURE_VERSION = 1`
- Persists: `config.json["feature_version"] == 1`
- Rejects: missing or unequal versions with `ValueError("校准格式已升级，请重新校准")`

- [ ] **Step 1: Write the failing legacy-config test**

Add `import json` near the other stdlib imports, then add:

```python
def test_calibration_rejects_legacy_feature_schema(self):
    image, rect = standard_board_image()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        Calibration.create(image, rect).save(directory)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("feature_version", None)
        config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "校准格式已升级"):
            Calibration.load(directory)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest -v test_assistant.RecognitionTests.test_calibration_rejects_legacy_feature_schema
```

Expected before the fix: `FAIL` because the loader currently accepts a config without `feature_version`.

- [ ] **Step 3: Implement the minimum feature-version contract**

Add near `STANDARD_BOARD`:

```python
CALIBRATION_FEATURE_VERSION = 1
```

Add to the JSON object in `Calibration.save()`:

```python
"feature_version": CALIBRATION_FEATURE_VERSION,
```

Immediately after reading `config.json` in `Calibration.load()`, add:

```python
if config.get("feature_version") != CALIBRATION_FEATURE_VERSION:
    raise ValueError("校准格式已升级，请重新校准")
```

Do not migrate unknown vectors in place. A one-time explicit recalibration is safer than guessing feature compatibility.

- [ ] **Step 4: Run persistence tests and verify GREEN**

Run:

```bash
python3 -m unittest -v \
  test_assistant.RecognitionTests.test_calibration_rejects_legacy_feature_schema \
  test_assistant.RecognitionTests.test_calibration_round_trip_preserves_recognition \
  test_assistant.RecognitionTests.test_calibration_rejects_corrupted_samples
```

Expected: all three pass. This task is not committed because both modified files contained user changes before this plan.

---

### Task 3: Identify the fixed Windows build and verify the full system

**Files:**
- Modify: `app.py` constant `APP_VERSION`
- Verify: `app.py`, `capture.py`, `board_recognition.py`, `pikafish_engine.py`, `game_state.py`, `test_assistant.py`

**Interfaces:**
- Produces: visible application version `2026.07.22.1`

- [ ] **Step 1: Update only the application version**

```python
APP_VERSION = "2026.07.22.1"
```

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
python3 -m unittest -v test_assistant.py
```

Expected: 110 tests run, 0 failures, with only the existing optional Tencent screenshot test skipped.

- [ ] **Step 3: Compile all production modules**

Run:

```bash
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Replay the supplied real snapshot with positive and negative drift**

Create one calibration from `analysis_snapshot.png`, translate the frame by `(-16, -16)` and `(16, 16)` without changing its dimensions, and call `recognize(translated_frame, align_grid=True)` for both.

Expected for the original and both translations:

```text
valid=True
board==STANDARD_BOARD
confidence>=0.45
```

- [ ] **Step 5: Review the incremental diff without overwriting existing work**

Run:

```bash
git diff --check
git status --short
git diff -- board_recognition.py test_assistant.py app.py
```

Separate pre-existing whitespace warnings from lines introduced by this implementation. Confirm `capture.py`, `pikafish_engine.py`, `game_state.py`, and `pikafish/` have no new task-related changes.

- [ ] **Step 6: Windows handoff**

On Windows 10, rebuild with:

```bat
build_exe.bat
```

Expected: `dist\XiangqiAssistant\XiangqiAssistant.exe` opens with title version `2026.07.22.1`. Because `feature_version` intentionally invalidates the old calibration once, perform one standard-opening calibration, then confirm “同步并分析” enters Pikafish analysis without the `0.29` error.

This Windows GUI/build step cannot be claimed from the non-Windows development host; report it as the only remaining manual verification boundary.
