import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import Mock, patch

import cv2
import numpy as np

import board_recognition as br
import app as app_module
from capture import parse_adb_devices, select_emulator_windows, validate_frame
from board_recognition import (
    STANDARD_BOARD,
    Calibration,
    Move,
    MotionTracker,
    board_to_fen,
    detect_move,
    detect_board_rect,
    infer_move_from_frames,
    grid_points,
    normalize_board,
    recognition_confidence,
)
from pikafish_engine import PikafishEngine
from app import AssistantApp, create_calibration, window_source_id


ROOT = Path(__file__).resolve().parent


def moved(board, start, end):
    result = [list(row) for row in board]
    result[end[0]][end[1]] = result[start[0]][start[1]]
    result[start[0]][start[1]] = ""
    return tuple(tuple(row) for row in result)


def standard_board_image(circular=True):
    image = np.full((780, 600, 3), 150, dtype=np.uint8)
    rect = (60, 65, 540, 695)
    points = grid_points(rect)
    for row in points:
        cv2.line(image, row[0], row[-1], (90, 90, 90), 2)
    for col in range(9):
        cv2.line(image, points[0][col], points[-1][col], (90, 90, 90), 2)
    for row_index, row in enumerate(STANDARD_BOARD):
        for col_index, piece in enumerate(row):
            if not piece:
                continue
            center = points[row_index][col_index]
            fill = (195, 220, 245) if piece.isupper() else (220, 205, 175)
            ink = (35, 35, 180) if piece.isupper() else (25, 25, 25)
            if circular:
                cv2.circle(image, center, 27, fill, -1)
                cv2.circle(image, center, 27, ink, 3)
            else:
                cv2.rectangle(
                    image, (center[0] - 24, center[1] - 24),
                    (center[0] + 24, center[1] + 24), fill, -1,
                )
                cv2.rectangle(
                    image, (center[0] - 24, center[1] - 24),
                    (center[0] + 24, center[1] + 24), ink, 3,
                )
            cv2.putText(
                image, piece.upper(), (center[0] - 10, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ink, 2, cv2.LINE_AA,
            )
    return image, rect


class CaptureTests(unittest.TestCase):
    def test_parse_adb_devices_keeps_only_online_devices(self):
        output = (
            "List of devices attached\n"
            "emulator-5554\tdevice\n"
            "127.0.0.1:5555\toffline\n"
            "ABC\tunauthorized\n"
        )
        self.assertEqual(parse_adb_devices(output), ["emulator-5554"])

    def test_blank_frame_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "空白"):
            validate_frame(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_changed_frame_size_is_rejected(self):
        frame = np.full((100, 100, 3), 127, dtype=np.uint8)
        frame[20:80, 20:80] = 200
        with self.assertRaisesRegex(RuntimeError, "尺寸"):
            validate_frame(frame, (120, 100))

    def test_jj_game_window_replaces_tencent_launcher(self):
        windows = [
            (100, "腾讯手游助手"),
            (200, "JJ象棋"),
            (300, "记事本"),
        ]
        self.assertEqual(select_emulator_windows(windows), [(200, "JJ象棋")])

    def test_ldplayer_window_remains_supported(self):
        windows = [(100, "雷电模拟器"), (300, "记事本")]
        self.assertEqual(select_emulator_windows(windows), [(100, "雷电模拟器")])


class BoardTests(unittest.TestCase):
    def test_piece_colors_infer_red_and_black_player_views(self):
        image, rect = standard_board_image()
        self.assertEqual(br.infer_player_from_colors(image, rect), "w")

        rotated = cv2.rotate(image, cv2.ROTATE_180)
        height, width = image.shape[:2]
        rotated_rect = (
            width - rect[2],
            height - rect[3],
            width - rect[0],
            height - rect[1],
        )
        self.assertEqual(br.infer_player_from_colors(rotated, rotated_rect), "b")

    def test_piece_colors_return_none_when_red_signal_is_ambiguous(self):
        image, rect = standard_board_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.assertIsNone(br.infer_player_from_colors(gray, rect))

    def test_standard_board_generates_initial_fen(self):
        self.assertEqual(
            board_to_fen(STANDARD_BOARD, "w"),
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )

    def test_grid_has_ninety_evenly_spaced_points(self):
        points = grid_points((36, 225, 511, 744))
        self.assertEqual((len(points), len(points[0])), (10, 9))
        self.assertEqual(points[0][0], (36, 225))
        self.assertEqual(points[-1][-1], (511, 744))

    def test_rotated_board_normalizes_to_standard(self):
        rotated = tuple(tuple(reversed(row)) for row in reversed(STANDARD_BOARD))
        self.assertEqual(normalize_board(rotated, True), STANDARD_BOARD)

    def test_fen_round_trip_preserves_board_and_active_side(self):
        board, side = br.fen_to_board(board_to_fen(STANDARD_BOARD, "b"))
        self.assertEqual(board, STANDARD_BOARD)
        self.assertEqual(side, "b")

    def test_fen_rejects_facing_kings(self):
        with self.assertRaisesRegex(ValueError, "将帅不能照面"):
            br.fen_to_board("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")

    def test_move_to_uci_uses_pikafish_coordinates(self):
        self.assertEqual(br.move_to_uci(Move("w", (9, 7), (7, 6))), "h0g2")


class RecognitionTests(unittest.TestCase):
    def test_piece_inventory_resolves_moved_cannon_visual_distractor(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        after = image.copy()

        def copy_cell(source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            after[dy - half : dy - half + size, dx - half : dx - half + size] = (
                image[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        copy_cell((7, 0), (7, 1))
        copy_cell((9, 1), (7, 4))
        classified = calibration.recognize(after)
        expected = moved(STANDARD_BOARD, (7, 1), (7, 4))
        self.assertTrue(classified.valid, classified.error)
        self.assertEqual(classified.board, expected)
        self.assertEqual(
            infer_move_from_frames(calibration, STANDARD_BOARD, image, after, "w"),
            expected,
        )

    def test_motion_tracker_requires_two_stable_frames(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        after = image.copy()

        for source, destination in (((7, 0), (7, 1)), ((7, 1), (7, 4))):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            after[dy - half : dy - half + size, dx - half : dx - half + size] = (
                image[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        tracker = MotionTracker(calibration)
        self.assertEqual(tracker.observe(image, "w").board, STANDARD_BOARD)
        self.assertEqual(tracker.observe(after, "w").board, STANDARD_BOARD)
        self.assertEqual(
            tracker.observe(after, "w").board,
            moved(STANDARD_BOARD, (7, 1), (7, 4)),
        )

    def test_motion_tracker_recovers_two_moves_from_one_frame(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        def paste(target, source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            target[dy - half : dy - half + size, dx - half : dx - half + size] = (
                image[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        final = image.copy()
        for start, end in (((9, 7), (7, 6)), ((0, 1), (2, 2))):
            paste(final, (5, 4), start)
            paste(final, start, end)

        after_red = moved(STANDARD_BOARD, (9, 7), (7, 6))
        after_both = moved(after_red, (0, 1), (2, 2))
        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")
        tracker.observe(final, "w")
        self.assertEqual(tracker.observe(final, "w").board, after_red)
        self.assertEqual(tracker.observe(final, "b").board, after_both)

    def test_motion_refreshes_single_faded_highlight_before_opponent_move(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        def copy_cell(target, base, source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            target[dy - half : dy - half + size, dx - half : dx - half + size] = (
                base[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        after_red = image.copy()
        copy_cell(after_red, image, (7, 0), (7, 1))
        copy_cell(after_red, image, (7, 1), (7, 4))
        highlighted = after_red.copy()
        cv2.circle(highlighted, points[7][1], 10, (0, 255, 0), 3)

        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")
        tracker.observe(highlighted, "w")
        accepted_red = tracker.observe(highlighted, "w")
        self.assertEqual(accepted_red.board, moved(STANDARD_BOARD, (7, 1), (7, 4)))
        tracker.observe(after_red, "b")
        tracker.observe(after_red, "b")

        after_black = after_red.copy()
        copy_cell(after_black, image, (1, 1), (0, 1))
        copy_cell(after_black, image, (0, 1), (2, 2))
        tracker.observe(after_black, "b")
        accepted_black = tracker.observe(after_black, "b")
        self.assertEqual(
            accepted_black.board,
            moved(accepted_red.board, (0, 1), (2, 2)),
        )

    def test_motion_does_not_accept_selected_piece_without_displacement(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        selected = image.copy()
        cv2.circle(selected, points[3][4], 14, (0, 255, 0), 3)
        tracker = MotionTracker(calibration)
        tracker.observe(image, "b")

        first = tracker.observe(selected, "b")
        second = tracker.observe(selected, "b")

        self.assertEqual(first.board, STANDARD_BOARD)
        self.assertEqual(second.board, STANDARD_BOARD)
        self.assertFalse(first.recovery)
        self.assertEqual(first.reason, "no_move")
        self.assertEqual(len(first.changes), 1)
        self.assertEqual(first.error, "检测到选中操作，等待棋子落点")

    def test_motion_accepts_opponent_move_when_highlights_switch_in_same_frame(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        def copy_cell(target, base, source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            target[dy - half : dy - half + size, dx - half : dx - half + size] = (
                base[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        after_red = image.copy()
        copy_cell(after_red, image, (5, 4), (7, 1))
        copy_cell(after_red, image, (7, 1), (7, 4))
        red_highlight = after_red.copy()
        cv2.circle(red_highlight, points[7][1], 12, (0, 255, 0), 3)
        cv2.circle(red_highlight, points[7][4], 12, (0, 255, 0), 3)

        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")
        tracker.observe(red_highlight, "w")
        accepted_red = tracker.observe(red_highlight, "w")

        after_black = after_red.copy()
        copy_cell(after_black, image, (5, 4), (3, 4))
        copy_cell(after_black, image, (3, 4), (4, 4))
        black_highlight = after_black.copy()
        cv2.circle(black_highlight, points[3][4], 12, (0, 255, 0), 3)
        cv2.circle(black_highlight, points[4][4], 12, (0, 255, 0), 3)

        tracker.observe(black_highlight, "b")
        accepted_black = tracker.observe(black_highlight, "b")

        self.assertEqual(
            accepted_black.board,
            moved(accepted_red.board, (3, 4), (4, 4)),
        )

    def test_motion_ignores_unchanged_frame(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        self.assertIsNone(
            infer_move_from_frames(
                calibration, STANDARD_BOARD, image, image.copy(), "w"
            )
        )

    def test_motion_reports_unresolved_board_change(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        changed = image.copy()
        points = grid_points(rect)
        cv2.circle(changed, points[5][4], 14, (0, 255, 0), -1)
        cv2.circle(changed, points[5][5], 14, (0, 255, 0), -1)
        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")

        result = tracker.observe(changed, "w")

        self.assertEqual(
            result.error,
            "棋盘变化无法组成当前方合法走法，正在恢复",
        )
        self.assertEqual(result.reason, "illegal")

    def test_motion_requests_recovery_after_three_unresolved_frames(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        recognized = moved(STANDARD_BOARD, (9, 7), (7, 6))
        calibration.recognize = Mock(
            side_effect=[
                br.Recognition(STANDARD_BOARD, 1.0, True),
                br.Recognition(recognized, 1.0, True),
            ]
        )
        changed = image.copy()
        points = grid_points(rect)
        cv2.circle(changed, points[5][4], 14, (0, 255, 0), -1)
        cv2.circle(changed, points[5][5], 14, (0, 255, 0), -1)
        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")

        self.assertFalse(tracker.observe(changed, "w").recovery)
        self.assertFalse(tracker.observe(changed, "w").recovery)
        result = tracker.observe(changed, "w")
        self.assertTrue(result.recovery)
        self.assertEqual(result.reason, "recovery_succeeded")
        self.assertEqual(result.board, recognized)

        tracker.reset(changed, recognized)
        self.assertEqual(tracker.board, recognized)
        self.assertEqual(tracker.unresolved_count, 0)

    def test_motion_requests_recovery_after_one_changed_cell_persists(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        calibration.recognize = Mock(
            side_effect=[
                br.Recognition(STANDARD_BOARD, 1.0, True),
                br.Recognition(STANDARD_BOARD, 1.0, True),
            ]
        )
        changed = image.copy()
        cv2.circle(changed, grid_points(rect)[5][4], 14, (0, 255, 0), -1)
        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")

        self.assertFalse(tracker.observe(changed, "w").recovery)
        self.assertFalse(tracker.observe(changed, "w").recovery)
        result = tracker.observe(changed, "w")

        self.assertTrue(result.recovery)
        self.assertEqual(result.reason, "recovery_succeeded")

    def test_motion_preserves_full_recognition_failure_reason(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        calibration.recognize = Mock(
            side_effect=[
                br.Recognition(STANDARD_BOARD, 1.0, True),
                br.Recognition(tuple(), 0.0, False, "置信度过低"),
            ]
        )
        changed = image.copy()
        points = grid_points(rect)
        cv2.circle(changed, points[5][4], 14, (0, 255, 0), -1)
        cv2.circle(changed, points[5][5], 14, (0, 255, 0), -1)
        tracker = MotionTracker(calibration)
        tracker.observe(image, "w")

        tracker.observe(changed, "w")
        tracker.observe(changed, "w")
        result = tracker.observe(changed, "w")

        self.assertEqual(
            result.error,
            "无法自动识别当前棋盘，请重新校准: 置信度过低",
        )
        self.assertEqual(result.reason, "recovery_failed")
        self.assertGreaterEqual(len(result.changes), 2)

    def test_motion_infers_capture(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        after = image.copy()
        for source, destination in (((7, 0), (7, 1)), ((7, 1), (0, 1))):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            after[dy - half : dy - half + size, dx - half : dx - half + size] = (
                image[sy - half : sy - half + size, sx - half : sx - half + size]
            )
        self.assertEqual(
            infer_move_from_frames(calibration, STANDARD_BOARD, image, after, "w"),
            moved(STANDARD_BOARD, (7, 1), (0, 1)),
        )

    def test_motion_detects_same_glyph_opposite_color_capture(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        reference = image.copy()

        def copy_cell(target, base, source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            target[dy - half : dy - half + size, dx - half : dx - half + size] = (
                base[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        copy_cell(reference, image, (5, 0), (3, 0))
        copy_cell(reference, image, (5, 0), (6, 0))
        board = [list(row) for row in STANDARD_BOARD]
        board[3][0] = board[6][0] = ""
        board = tuple(tuple(row) for row in board)
        after = reference.copy()
        copy_cell(after, reference, (5, 0), (9, 0))
        copy_cell(after, reference, (9, 0), (0, 0))

        self.assertEqual(
            infer_move_from_frames(calibration, board, reference, after, "w"),
            moved(board, (9, 0), (0, 0)),
        )

    def test_motion_ignores_third_changed_cell_when_legal_move_is_unique(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        after = image.copy()

        def copy_cell(source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            after[dy - half : dy - half + size, dx - half : dx - half + size] = (
                image[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        copy_cell((7, 0), (7, 1))
        copy_cell((7, 1), (7, 4))
        copy_cell((9, 0), (5, 5))
        self.assertEqual(
            infer_move_from_frames(calibration, STANDARD_BOARD, image, after, "w"),
            moved(STANDARD_BOARD, (7, 1), (7, 4)),
        )

    def test_motion_ignores_small_global_brightness_change(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        brighter = cv2.convertScaleAbs(image, alpha=1.0, beta=5)
        self.assertIsNone(
            infer_move_from_frames(
                calibration, STANDARD_BOARD, image, brighter, "w"
            )
        )

    def test_motion_normalizes_rotated_screen_coordinates(self):
        image, rect = standard_board_image()
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        h, w = image.shape[:2]
        rotated_rect = (w - rect[2], h - rect[3], w - rect[0], h - rect[1])
        calibration = Calibration.create(rotated, rotated_rect, True)
        points = grid_points(rotated_rect)
        size, half = calibration.size, calibration.size // 2
        after = rotated.copy()
        for source, destination in (((2, 8), (2, 7)), ((2, 7), (2, 4))):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            after[dy - half : dy - half + size, dx - half : dx - half + size] = (
                rotated[sy - half : sy - half + size, sx - half : sx - half + size]
            )
        self.assertEqual(
            infer_move_from_frames(calibration, STANDARD_BOARD, rotated, after, "w"),
            moved(STANDARD_BOARD, (7, 1), (7, 4)),
        )

    def test_calibration_recognizes_supplied_initial_board(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        result = calibration.recognize(image)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)
        self.assertEqual(sum(bool(piece) for row in result.board for piece in row), 32)

    @unittest.skipUnless((ROOT / "123.jpg").is_file(), "未提供腾讯截图样本")
    def test_calibration_recognizes_tencent_jj_window_screenshot(self):
        image = cv2.imread(str(ROOT / "123.jpg"))
        calibration = Calibration.create(image, (27, 220, 413, 642))
        result = calibration.recognize(image)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_calibration_round_trip_preserves_recognition(self):
        image, rect = standard_board_image()
        with tempfile.TemporaryDirectory() as tmp:
            Calibration.create(image, rect).save(Path(tmp))
            result = Calibration.load(Path(tmp)).recognize(image)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_calibration_recognizes_piece_moved_to_new_intersection(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(calibration.rect)
        size, half = calibration.size, calibration.size // 2
        moved_image = image.copy()

        def crop(row, col):
            x, y = points[row][col]
            return image[y - half : y - half + size, x - half : x - half + size].copy()

        def paste(row, col, value):
            x, y = points[row][col]
            moved_image[y - half : y - half + size, x - half : x - half + size] = value

        paste(7, 1, crop(7, 0))
        paste(7, 4, crop(7, 1))
        result = calibration.recognize(moved_image)
        self.assertTrue(result.valid, result.error)
        self.assertEqual((result.board[7][1], result.board[7][4]), ("", "C"))

    def test_full_recognition_limits_low_confidence_duplicate_piece(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        with tempfile.TemporaryDirectory() as tmp:
            calibration.save(Path(tmp))
            calibration = Calibration.load(Path(tmp))
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        def crop(row, col):
            x, y = points[row][col]
            return image[
                y - half : y - half + size,
                x - half : x - half + size,
            ].copy()

        current = image.copy()
        x, y = points[5][5]
        current[y - half : y - half + size, x - half : x - half + size] = (
            cv2.addWeighted(crop(7, 1), 0.8, crop(5, 5), 0.2, 0)
        )

        result = calibration.recognize(current)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_full_recognition_ignores_small_cursor_circle_on_empty_point(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        current = image.copy()
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        empty_x, empty_y = points[5][5]
        target_x, target_y = points[0][5]
        current[
            target_y - half : target_y - half + size,
            target_x - half : target_x - half + size,
        ] = image[
            empty_y - half : empty_y - half + size,
            empty_x - half : empty_x - half + size,
        ]
        cv2.circle(
            current,
            points[0][5],
            10,
            (255, 255, 255),
            3,
        )
        expected = [list(row) for row in STANDARD_BOARD]
        expected[0][5] = ""
        expected = tuple(tuple(row) for row in expected)

        result = calibration.recognize(current)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, expected)

    def test_full_recognition_never_assigns_piece_without_large_circle(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label and (row, col) != (0, 5)
        }
        # (0, 5) is black advisor 'a'. Since occupied excludes (0, 5) and the bypass threshold is 0.75,
        # 'a' at (0, 5) will be correctly cleared/not assigned since it misses the occupancy circle.
        expected = [list(row) for row in STANDARD_BOARD]
        expected[0][5] = ""
        expected = tuple(tuple(row) for row in expected)
        calibration.threshold = 0.9

        with patch.object(
            calibration,
            "_occupied_cells",
            return_value=occupied,
            create=True,
        ):
            result = calibration.recognize(image)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, expected)

    def test_empty_cursor_marks_do_not_lower_piece_confidence(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        current = image.copy()
        points = grid_points(rect)
        marked_empty_cells = (
            (4, 0), (4, 2), (4, 4), (4, 6),
            (4, 8), (5, 1), (5, 3), (5, 5),
        )
        for row, col in marked_empty_cells:
            cv2.circle(current, points[row][col], 10, (255, 255, 255), 3)

        result = calibration.recognize(current)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_circle_occupancy_does_not_depend_on_empty_template_scores(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        calibration.samples[calibration.labels == ""] *= -1

        result = calibration.recognize(image)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_circle_board_uses_structure_when_absolute_scores_are_low(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        calibration.samples *= 0.4
        calibration.circle_occupancy = False

        result = calibration.recognize(image)

        self.assertLess(result.confidence, calibration.threshold)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_circle_board_still_rejects_missing_king(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label and (row, col) != (0, 4)
        }

        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            calibration.threshold = 0.9
            result = calibration.recognize(image)

        self.assertFalse(result.valid)
        self.assertIn("红帅或黑将数量错误", result.error)

    def test_old_circle_calibration_recovers_with_one_missed_piece(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        calibration.circle_occupancy = False
        calibration.samples *= 0.4
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label and (row, col) != (0, 5)
        }

        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            result = calibration.recognize(image)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[0][5], "")

    def test_recognized_red_and_black_moves_trigger_second_advice(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        def copy_cell(target, source_image, source, destination):
            sx, sy = points[source[0]][source[1]]
            dx, dy = points[destination[0]][destination[1]]
            target[dy - half : dy - half + size, dx - half : dx - half + size] = (
                source_image[sy - half : sy - half + size, sx - half : sx - half + size]
            )

        after_red = image.copy()
        copy_cell(after_red, image, (7, 0), (7, 1))
        copy_cell(after_red, image, (7, 1), (7, 4))
        after_black = after_red.copy()
        copy_cell(after_black, image, (1, 1), (0, 1))
        copy_cell(after_black, image, (0, 1), (2, 2))

        recognitions = [calibration.recognize(frame) for frame in (image, after_red, after_black)]
        self.assertTrue(all(result.valid for result in recognitions))
        class StopEvent:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, _seconds):
                return self.stopped

        frames = iter(frame for frame in (image, after_red, after_black) for _ in range(2))
        app = object.__new__(AssistantApp)
        app.stop_event = StopEvent()
        app.results = Queue()
        app.engine_lock = threading.Lock()
        app.engine = None
        app.calibration = calibration

        def capture():
            try:
                return next(frames)
            except StopIteration:
                app.stop_event.set()
                return after_black

        searches = []
        advice = iter((("h2e2", "+0.20"), ("b0c2", "+0.20")))
        app.capture = capture
        app.best_move = (
            lambda position, movetime=5000:
            searches.append(position) or next(advice)
        )
        app.run_loop("w")
        events = list(app.results.queue)
        self.assertEqual(len(searches), 2)
        self.assertTrue(searches[0].startswith("position fen "))
        # Both searches receive current board FEN (not moves history)
        self.assertNotIn(" moves ", searches[1], searches[1])
        self.assertTrue(searches[1].startswith("position fen "))
        self.assertEqual(sum(event[0] == "move" for event in events), 2)
        self.assertTrue(any(event[0] == "clear_move" for event in events))
        turns = [event[1] for event in events if event[0] == "turn"]
        self.assertEqual(turns, ["w", "b", "w"])

    def test_rotated_calibration_normalizes_screen_orientation(self):
        image, rect = standard_board_image()
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        h, w = image.shape[:2]
        rotated_rect = (w - rect[2], h - rect[3], w - rect[0], h - rect[1])
        result = Calibration.create(rotated, rotated_rect, True).recognize(rotated)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_calibration_persists_source_and_rejects_new_frame_size(self):
        image, rect = standard_board_image()
        with tempfile.TemporaryDirectory() as tmp:
            Calibration.create(
                image, rect, source_id="ADB · emulator-5554"
            ).save(Path(tmp))
            calibration = Calibration.load(Path(tmp))
        self.assertEqual(calibration.source_id, "ADB · emulator-5554")
        resized = cv2.resize(image, (image.shape[1] + 10, image.shape[0]))
        result = calibration.recognize(resized)
        self.assertFalse(result.valid)
        self.assertIn("截图尺寸已变化", result.error)

    def test_calibration_rejects_corrupted_samples(self):
        image, rect = standard_board_image()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Calibration.create(image, rect).save(directory)
            (directory / "samples.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "校准样本"):
                Calibration.load(directory)

    def test_automatic_calibration_finds_standard_board(self):
        image, expected = standard_board_image()
        rect = detect_board_rect(image)
        self.assertTrue(all(abs(a - b) <= 5 for a, b in zip(rect, expected)), rect)
        self.assertEqual(Calibration.create(image, rect).recognize(image).board, STANDARD_BOARD)

    def test_calibration_uses_manual_rect_only_after_automatic_failure(self):
        image, expected = standard_board_image()
        calls = []
        calibration, mode = create_calibration(
            image, False, "test", lambda _error: calls.append(_error) or expected
        )
        self.assertEqual(mode, "自动")
        self.assertEqual(calls, [])
        
        # ponytail: Mock detect_board_rect to raise ValueError to simulate automatic detection failure,
        # forcing the create_calibration function to fall back to manual rect path safely.
        with patch("app.detect_board_rect", side_effect=ValueError("自动检测失败")):
            calibration, mode = create_calibration(
                image, False, "test", lambda error: calls.append(error) or expected
            )
        self.assertEqual(mode, "手动")
        self.assertTrue(calls)

    def test_confidence_ignores_two_changed_intersections(self):
        scores = [0.82] * 88 + [0.11, 0.19]
        self.assertGreater(recognition_confidence(scores), 0.45)


class MoveTests(unittest.TestCase):
    def test_detect_move_handles_quiet_move(self):
        after = moved(STANDARD_BOARD, (7, 1), (7, 4))
        self.assertEqual(detect_move(STANDARD_BOARD, after), Move("w", (7, 1), (7, 4)))

    def test_detect_move_handles_cannon_capture(self):
        after = moved(STANDARD_BOARD, (7, 1), (0, 1))
        self.assertEqual(detect_move(STANDARD_BOARD, after), Move("w", (7, 1), (0, 1)))

    def test_detect_move_rejects_rook_jumping_over_piece(self):
        after = moved(STANDARD_BOARD, (9, 0), (4, 0))
        self.assertIsNone(detect_move(STANDARD_BOARD, after))

    def test_detect_move_rejects_advisor_outside_palace(self):
        after = moved(STANDARD_BOARD, (9, 3), (8, 2))
        self.assertIsNone(detect_move(STANDARD_BOARD, after))

class GameStateTests(unittest.TestCase):
    def test_records_actual_moves_and_restores_history(self):
        from game_state import GameState

        state = GameState.start(STANDARD_BOARD, "w", "w")
        after_red = moved(STANDARD_BOARD, (9, 7), (7, 6))
        self.assertTrue(state.apply_board(after_red))
        self.assertEqual(state.moves, ["h0g2"])
        self.assertEqual(state.side_to_move, "b")
        self.assertEqual(state.position_version, 1)

        after_black = moved(after_red, (0, 1), (2, 2))
        self.assertTrue(state.apply_board(after_black))
        version = state.position_version
        self.assertTrue(state.restore_snapshot(STANDARD_BOARD))
        self.assertEqual(state.moves, [])
        self.assertEqual(state.side_to_move, "w")
        self.assertGreater(state.position_version, version)

    def test_repeated_board_is_analyzed_in_new_version(self):
        from game_state import GameState

        state = GameState.start(STANDARD_BOARD, "w", "w")
        self.assertTrue(state.needs_analysis())
        state.mark_analyzed()
        self.assertFalse(state.needs_analysis())
        after_red = moved(STANDARD_BOARD, (9, 7), (7, 6))
        state.apply_board(after_red)
        state.restore_snapshot(STANDARD_BOARD)
        self.assertTrue(state.needs_analysis())

    def test_resync_starts_new_history_base(self):
        from game_state import GameState

        state = GameState.start(STANDARD_BOARD, "w", "w")
        after_red = moved(STANDARD_BOARD, (9, 7), (7, 6))
        state.apply_board(after_red)
        state.resync(after_red, "b")
        self.assertEqual(state.moves, [])
        self.assertEqual(state.side_to_move, "b")
        self.assertEqual(state.position_command(), f"position fen {board_to_fen(after_red, 'b')}")


class EngineTests(unittest.TestCase):
    def test_chinese_move_uses_board_before_move(self):
        self.assertEqual(
            PikafishEngine.get_chinese_move("h2e2", STANDARD_BOARD), "炮二平五"
        )

    def test_chinese_move_tracks_cannon_after_it_reaches_center_file(self):
        centered = moved(STANDARD_BOARD, (7, 1), (7, 4))
        self.assertEqual(
            PikafishEngine.get_chinese_move("e2e6", centered), "炮五进四"
        )

    def test_read_until_has_finite_timeout(self):
        engine = object.__new__(PikafishEngine)
        engine._lines = Queue()
        with self.assertRaises(TimeoutError):
            engine._read_until(lambda line: line == "uciok", 0.01)

    def test_read_until_preserves_engine_exit_diagnostics(self):
        engine = object.__new__(PikafishEngine)
        engine._lines = Queue()
        engine.process = Mock()
        engine.process.poll.return_value = -1073741795
        engine._lines.put("info string NNUE load failed")
        engine._lines.put(None)

        with self.assertRaisesRegex(
            RuntimeError,
            r"退出码.*0xC000001D.*NNUE load failed",
        ):
            engine._read_until(lambda line: line == "uciok", 1.0)

    def test_configure_strength_uses_all_cpu_and_large_hash(self):
        engine = object.__new__(PikafishEngine)
        sent = []
        engine.send_command = sent.append
        uci_lines = [
            "option name Threads type spin default 1 min 1 max 1024",
            "option name Hash type spin default 16 min 1 max 33554432",
            "option name MultiPV type spin default 1 min 1 max 128",
        ]

        with patch("pikafish_engine.os.cpu_count", return_value=12):
            engine._configure_strength(uci_lines)

        self.assertEqual(
            sent,
            [
                "setoption name Threads value 12",
                "setoption name Hash value 128",
                "setoption name MultiPV value 1",
            ],
        )

    def test_engine_uses_complete_position_history(self):
        engine = object.__new__(PikafishEngine)
        engine._lock = threading.Lock()
        sent = []
        engine.send_command = sent.append
        engine._read_until = lambda _predicate, _timeout: ["bestmove a0a1"]
        command = f"position fen {board_to_fen(STANDARD_BOARD, 'w')} moves h0g2"

        engine.get_best_move(command, movetime=5000)

        # Engine must receive the full command including moves history (not stripped)
        self.assertEqual(sent, ["ucinewgame", "isready", command, "go movetime 5000"])

    def test_windows_engine_process_has_no_console(self):
        engine = object.__new__(PikafishEngine)
        engine.binary_path = "pikafish.exe"
        engine.process = None
        engine._lines = Queue()
        engine._lock = threading.Lock()
        process = Mock(stdout=[])
        process.poll.return_value = None
        no_window = 0x08000000

        with (
            patch("pikafish_engine.os.path.isfile", return_value=True),
            patch("pikafish_engine.platform.system", return_value="Windows"),
            patch.object(subprocess, "CREATE_NO_WINDOW", no_window, create=True),
            patch("pikafish_engine.subprocess.Popen", return_value=process) as popen,
            patch("pikafish_engine.threading.Thread"),
            patch.object(engine, "_read_until", return_value=[]),
        ):
            engine._start(1.0)

        self.assertEqual(popen.call_args.kwargs["creationflags"], no_window)

    def test_close_ignores_invalid_windows_pipe_handle(self):
        engine = object.__new__(PikafishEngine)
        engine._lock = threading.Lock()
        process = Mock()
        process.poll.return_value = None
        process.stdin.write.side_effect = OSError(22, "Invalid argument")
        process.stdout.close.side_effect = OSError(22, "Invalid argument")
        engine.process = process

        engine.close()

        self.assertIsNone(engine.process)
        process.kill.assert_called_once()

    def test_best_move_retries_engine_start_after_errno_22(self):
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.engine = None
        app.stop_event = Mock(is_set=Mock(return_value=False))
        app.closing = False
        healthy = Mock()
        healthy.get_best_move.return_value = ("b2e2", "+0.20")

        with patch(
            "app.PikafishEngine",
            side_effect=[OSError(22, "Invalid argument"), healthy],
        ):
            result = app.best_move("position fen test", 1000)

        self.assertEqual(result, ("b2e2", "+0.20"))

    def test_best_advice_restarts_after_illegal_elephant_move(self):
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.engine = Mock()
        app.best_move = Mock(
            side_effect=[("c0e0", "-9.99"), ("h2e2", "+0.20")]
        )

        result = app.best_advice(
            STANDARD_BOARD,
            f"position fen {board_to_fen(STANDARD_BOARD, 'w')}",
            "w",
            1000,
        )

        self.assertEqual(result, ("h2e2", "+0.20", "炮二平五"))
        self.assertEqual(app.best_move.call_count, 2)

    @unittest.skipIf(os.name == "nt", "临时POSIX可执行脚本测试")
    def test_startup_timeout_reaps_process(self):
        engine = object.__new__(PikafishEngine)
        engine.binary_path = "/bin/cat"
        engine.process = None
        engine._lines = Queue()
        engine._lock = threading.Lock()
        try:
            with self.assertRaises(TimeoutError):
                engine._start(0.05)
            self.assertTrue(
                engine.process is None or engine.process.poll() is not None,
                "启动超时后Pikafish子进程仍在运行",
            )
        finally:
            engine.close()

    def test_invalid_uci_coordinates_return_none(self):
        self.assertIsNone(PikafishEngine.uci_to_coords("abcd"))

    def test_real_engine_returns_legal_uci_move(self):
        engine = PikafishEngine()
        try:
            move, _ = engine.get_best_move(
                board_to_fen(STANDARD_BOARD, "w"), movetime=200
            )
        finally:
            engine.close()
        self.assertRegex(move, r"^[a-i][0-9][a-i][0-9]$")

    def test_real_engine_analyzes_reported_midgame_position(self):
        fen = (
            "1nbak2r1/c3a4/r2cb1nC1/p1p1C1pRp/9/4P1P2/"
            "P1P5P/2N3N2/4A4/2BAK1B2 w - - 0 1"
        )
        engine = PikafishEngine()
        try:
            move, _ = engine.get_best_move(fen, movetime=200)
        finally:
            engine.close()
        self.assertRegex(move, r"^[a-i][0-9][a-i][0-9]$")


class LoopTests(unittest.TestCase):
    def test_defaults_use_ten_second_search_and_point_two_sampling(self):
        self.assertEqual(app_module.DEFAULT_SEARCH_SECONDS, 10)
        self.assertEqual(app_module.POLL_INTERVAL_SECONDS, 0.2)

    def test_analyze_snapshot_calls_engine_once_and_stops(self):
        app = object.__new__(AssistantApp)
        app.stop_event = threading.Event()
        app.results = Queue()
        app.best_move = Mock(return_value=("b2e2", "+0.20"))

        app.analyze_snapshot(
            STANDARD_BOARD,
            board_to_fen(STANDARD_BOARD, "w"),
            1000,
        )

        app.best_move.assert_called_once_with(
            board_to_fen(STANDARD_BOARD, "w"), 1000
        )
        events = list(app.results.queue)
        self.assertEqual(events[0], ("move", "炮八平五", "b2e2 · +0.20"))
        self.assertEqual(events[-1], ("stopped",))

    def test_analyze_snapshot_drops_result_after_stop(self):
        app = object.__new__(AssistantApp)
        app.stop_event = threading.Event()
        app.results = Queue()

        def stopped_search(_position, _movetime):
            app.stop_event.set()
            return "b2e2", "+0.20"

        app.best_move = stopped_search
        app.analyze_snapshot(
            STANDARD_BOARD,
            board_to_fen(STANDARD_BOARD, "w"),
            1000,
        )

        self.assertEqual(list(app.results.queue), [("stopped",)])

    def test_diagnostic_frames_overwrite_one_latest_pair(self):
        first = np.zeros((10, 10, 3), dtype=np.uint8)
        second = np.full((10, 10, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            app_module.save_diagnostic_frames(Path(tmp), first, second)
            app_module.save_diagnostic_frames(Path(tmp), second, first)
            files = sorted(
                path.name for path in Path(tmp).glob("diagnostic_*.png")
            )
        self.assertEqual(
            files,
            ["diagnostic_current.png", "diagnostic_reference.png"],
        )

    def test_analysis_snapshot_overwrites_latest_frame(self):
        first = np.zeros((10, 10, 3), dtype=np.uint8)
        second = np.full((10, 10, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            app_module.save_analysis_snapshot(directory, first)
            app_module.save_analysis_snapshot(directory, second)
            saved = cv2.imread(str(directory / "analysis_snapshot.png"))
        self.assertTrue(np.array_equal(saved, second))

    def test_load_diagnostic_images_handles_valid_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            valid = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(directory / "analysis_snapshot.png"), valid)
            items = app_module.load_diagnostic_images(directory, (40, 40))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["title"], "最近一次同步分析快照")
            self.assertEqual(items[0]["image"].size, (40, 20))
            self.assertEqual(items[0]["error"], "")

            (directory / "analysis_snapshot.png").write_bytes(b"broken")
            items = app_module.load_diagnostic_images(directory, (40, 40))
            self.assertIn("无法读取图片", items[0]["error"])
            self.assertIsNone(items[0]["image"])

    @patch("app.messagebox.showinfo")
    def test_open_diagnostic_directory_creates_and_shows_path_without_startfile(
        self, showinfo
    ):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "diagnostics"
            app = object.__new__(AssistantApp)
            app.config_dir = directory
            with patch.object(app_module.os, "startfile", None, create=True):
                app.open_diagnostic_directory()
            self.assertTrue(directory.is_dir())
            showinfo.assert_called_once_with("诊断目录", str(directory))

    def test_open_diagnostic_directory_uses_windows_startfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "diagnostics"
            app = object.__new__(AssistantApp)
            app.config_dir = directory
            with patch.object(
                app_module.os, "startfile", create=True
            ) as startfile:
                app.open_diagnostic_directory()
            startfile.assert_called_once_with(str(directory))

    @patch("app.save_diagnostic_frames")
    @patch("app.MotionTracker")
    def test_run_loop_saves_frames_after_recovery_failure(
        self, tracker_class, save_frames
    ):
        from game_state import GameState

        class StopAfterWait:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, _seconds):
                self.stopped = True
                return True

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        motion = tracker_class.return_value
        motion.reference = frame.copy()
        motion.observe.return_value = br.Recognition(
            STANDARD_BOARD,
            0.0,
            True,
            "无法自动识别当前棋盘",
            reason="recovery_failed",
        )
        app = object.__new__(AssistantApp)
        app.stop_event = StopAfterWait()
        app.results = Queue()
        app.engine_lock = threading.Lock()
        app.engine = None
        app.calibration = Mock()
        app.config_dir = Path("diagnostics")
        app.capture = Mock(return_value=frame)
        state = GameState.start(STANDARD_BOARD, "w", "b")

        app.run_loop("b", state, frame)

        save_frames.assert_called_once_with(
            app.config_dir, motion.reference, frame
        )

    @patch("app.save_diagnostic_frames")
    @patch("app.MotionTracker")
    def test_run_loop_saves_frames_after_successful_recovery(
        self, tracker_class, save_frames
    ):
        from game_state import GameState

        class StopAfterWait:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, _seconds):
                self.stopped = True
                return True

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        motion = tracker_class.return_value
        motion.reference = frame.copy()
        motion.observe.return_value = br.Recognition(
            STANDARD_BOARD,
            1.0,
            True,
            "恢复诊断",
            recovery=True,
            reason="recovery_succeeded",
        )
        app = object.__new__(AssistantApp)
        app.stop_event = StopAfterWait()
        app.results = Queue()
        app.engine_lock = threading.Lock()
        app.engine = None
        app.calibration = Mock()
        app.config_dir = Path("diagnostics")
        app.capture = Mock(return_value=frame)
        state = GameState.start(STANDARD_BOARD, "w", "b")

        app.run_loop("b", state, frame)

        save_frames.assert_called_once_with(
            app.config_dir, motion.reference, frame
        )
        self.assertEqual(motion.reset.call_count, 2)
        motion.reset.assert_called_with(frame, STANDARD_BOARD)
        self.assertFalse(any(event[0] == "needs_sync" for event in app.results.queue))

    @patch("app.save_diagnostic_frames")
    @patch("app.MotionTracker")
    def test_run_loop_applies_recovered_legal_next_board(
        self, tracker_class, _save_frames
    ):
        from game_state import GameState

        class StopAfterWait:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, _seconds):
                self.stopped = True
                return True

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        recovered = moved(STANDARD_BOARD, (7, 1), (7, 4))
        motion = tracker_class.return_value
        motion.reference = frame.copy()
        motion.observe.return_value = br.Recognition(
            recovered,
            1.0,
            True,
            "已识别当前棋盘，正在恢复",
            recovery=True,
            reason="recovery_succeeded",
            changes=(((7, 1), 1.0),),
        )
        app = object.__new__(AssistantApp)
        app.stop_event = StopAfterWait()
        app.results = Queue()
        app.engine_lock = threading.Lock()
        app.engine = None
        app.calibration = Mock()
        app.config_dir = Path("diagnostics")
        app.capture = Mock(return_value=frame)
        app.best_move = Mock(return_value=("b9c7", "+0.20"))
        state = GameState.start(STANDARD_BOARD, "w", "b")

        app.run_loop("b", state, frame)

        self.assertEqual(state.board, recovered)
        self.assertEqual(state.moves, ["b2e2"])
        self.assertEqual(state.side_to_move, "b")
        self.assertEqual(state.position_version, 1)
        self.assertEqual(motion.reset.call_count, 2)
        motion.reset.assert_called_with(frame, recovered)
        self.assertTrue(any(event[0] == "clear_move" for event in app.results.queue))
        self.assertFalse(any(event[0] == "needs_sync" for event in app.results.queue))

    def test_movetime_accepts_only_one_to_sixty_seconds(self):
        self.assertEqual(app_module.parse_movetime_seconds("5"), 5000)
        for value in ("0", "61", "x", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app_module.parse_movetime_seconds(value)

    def test_poll_results_updates_live_side_to_move(self):
        app = object.__new__(AssistantApp)
        app.results = Queue()
        app.results.put(("turn", "b"))
        app.side_to_move_var = Mock()
        app.move_var = Mock()
        app.status_var = Mock()
        app.start_button = Mock()
        app.root = Mock()

        app.poll_results()

        app.side_to_move_var.set.assert_called_once_with("黑方")

    def test_poll_results_keeps_move_and_reenables_manual_sync(self):
        app = object.__new__(AssistantApp)
        app.results = Queue()
        app.results.put(("move", "炮八平五", "b2e2 · +0.20"))
        app.results.put(("stopped",))
        app.worker = Mock()
        app.move_var = Mock()
        app.status_var = Mock()
        app.sync_button = Mock()
        app.start_button = Mock()
        app.player_box = Mock()
        app.side_to_move_box = Mock()
        app.root = Mock()

        app.poll_results()

        app.move_var.set.assert_called_once_with("炮八平五")
        app.status_var.set.assert_called_once_with("b2e2 · +0.20")
        app.sync_button.configure.assert_called_once_with(state="normal")
        app.start_button.configure.assert_called_once_with(
            text="停止分析", state="disabled"
        )
        self.assertIsNone(app.worker)

    @patch("app.infer_player_from_colors", return_value="b", create=True)
    @patch("app.Calibration.create")
    @patch("app.create_calibration")
    def test_calibrate_automatically_updates_black_player_view(
        self, create_calibration_mock, calibration_create, _infer_player
    ):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first = Mock(rect=(10, 10, 90, 90), rotated=False)
        corrected = Mock(rect=first.rect, rotated=True)
        create_calibration_mock.return_value = (first, "自动")
        calibration_create.return_value = corrected

        app = object.__new__(AssistantApp)
        app.worker = None
        app.capture = Mock(return_value=frame)
        app.source_id = Mock(return_value="window:test")
        app.choose_board_rect = Mock()
        app.player_var = Mock()
        app.player_var.get.return_value = "红方"
        app.side_to_move_var = Mock()
        app.move_var = Mock()
        app.status_var = Mock()
        app.config_dir = Path("unused")

        app.calibrate()

        app.player_var.set.assert_called_once_with("黑方")
        app.side_to_move_var.set.assert_called_once_with("黑方")
        calibration_create.assert_called_once_with(
            frame, first.rect, True, "window:test"
        )
        corrected.save.assert_called_once_with(app.config_dir)
        self.assertIs(app.calibration, corrected)

    def _manual_sync_app(self, player="红方"):
        app = object.__new__(AssistantApp)
        app.worker = None
        app.calibration = Mock(
            rotated=player == "黑方",
            source_id="",
        )
        app.source_var = Mock()
        app.source_var.get.return_value = "window:test"
        app.sources = {"window:test": object()}
        app.source_id = Mock(return_value="window:test")
        app.player_var = Mock()
        app.player_var.get.return_value = player
        app.movetime_var = Mock()
        app.movetime_var.get.return_value = "1"
        app.config_dir = Path("diagnostics")
        app.stop_event = threading.Event()
        app.side_to_move_var = Mock()
        app.move_var = Mock()
        app.status_var = Mock()
        app.sync_button = Mock()
        app.start_button = Mock()
        app.player_box = Mock()
        app.side_to_move_box = Mock()
        app.analyze_snapshot = Mock()
        return app

    @patch("app.threading.Thread")
    @patch("app.save_analysis_snapshot")
    def test_sync_current_board_captures_saves_and_starts_analysis(
        self, save_snapshot, thread_class
    ):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        app = self._manual_sync_app()
        app.capture = Mock(return_value=frame)
        app.calibration.recognize.return_value = br.Recognition(
            STANDARD_BOARD, 1.0, True
        )

        app.sync_current_board()

        app.capture.assert_called_once_with()
        save_snapshot.assert_called_once_with(app.config_dir, frame)
        app.calibration.recognize.assert_called_once_with(frame)
        kwargs = thread_class.call_args.kwargs
        self.assertIs(kwargs["target"], app.analyze_snapshot)
        self.assertEqual(kwargs["args"][0], STANDARD_BOARD)
        self.assertEqual(kwargs["args"][1], board_to_fen(STANDARD_BOARD, "w"))
        self.assertEqual(kwargs["args"][2], 1000)
        thread_class.return_value.start.assert_called_once_with()
        app.move_var.set.assert_called_once_with("--")
        app.side_to_move_var.set.assert_called_once_with("红方")

    @patch("app.threading.Thread")
    @patch("app.save_analysis_snapshot")
    def test_sync_current_board_uses_black_player_as_active_side(
        self, _save_snapshot, thread_class
    ):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        app = self._manual_sync_app("黑方")
        app.capture = Mock(return_value=frame)
        app.calibration.recognize.return_value = br.Recognition(
            STANDARD_BOARD, 1.0, True
        )

        app.sync_current_board()

        self.assertEqual(
            thread_class.call_args.kwargs["args"][1],
            board_to_fen(STANDARD_BOARD, "b"),
        )
        app.side_to_move_var.set.assert_called_once_with("黑方")

    @patch("app.threading.Thread")
    @patch("app.save_analysis_snapshot")
    @patch("app.messagebox.showerror")
    def test_sync_current_board_recognition_failure_does_not_start_worker(
        self, showerror, save_snapshot, thread_class
    ):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        app = self._manual_sync_app()
        app.capture = Mock(return_value=frame)
        app.calibration.recognize.return_value = br.Recognition(
            tuple(), 0.0, False, "置信度过低"
        )

        app.sync_current_board()

        save_snapshot.assert_called_once_with(app.config_dir, frame)
        thread_class.assert_not_called()
        showerror.assert_called_once()
        app.status_var.set.assert_called_with(
            "无法识别当前棋盘: 置信度过低。请等待动画结束后重试；"
            "仅在分辨率、棋盘皮肤或执棋方向改变时重新校准"
        )

    @patch("app.messagebox.showinfo")
    def test_sync_current_board_rejects_second_click_while_worker_alive(
        self, _showinfo
    ):
        app = self._manual_sync_app()
        app.worker = Mock()
        app.worker.is_alive.return_value = True
        app.capture = Mock()

        app.sync_current_board()

        app.capture.assert_not_called()
        app.status_var.set.assert_called_once_with(
            "正在分析，请稍候或点击“停止分析”"
        )

    def test_close_stops_worker_and_engine_before_destroying_window(self):
        events = []

        class Worker:
            alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout):
                events.append(("join", timeout))
                self.alive = False

        class Engine:
            def close(self):
                events.append("engine")

        class Root:
            def destroy(self):
                events.append("destroy")

        app = object.__new__(AssistantApp)
        app.stop_event = threading.Event()
        app.worker = Worker()
        app.engine = Engine()
        app.engine_lock = threading.Lock()
        app.closing = False
        app.root = Root()
        app.close()
        self.assertTrue(app.stop_event.is_set())
        self.assertEqual(events[0], "engine")
        self.assertEqual(events[1][0], "join")
        self.assertEqual(events[-1], "destroy")

    def test_duplicate_window_titles_have_distinct_calibration_ids(self):
        self.assertEqual(window_source_id("雷电模拟器", 100, 1), "window:雷电模拟器")
        self.assertNotEqual(
            window_source_id("雷电模拟器", 100, 2),
            window_source_id("雷电模拟器", 200, 2),
        )


if __name__ == "__main__":
    unittest.main()
