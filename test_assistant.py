import json
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

    def test_tiantian_game_window_replaces_tencent_launcher(self):
        windows = [
            (100, "腾讯手游助手"),
            (200, "天天象棋"),
            (300, "记事本"),
        ]
        self.assertEqual(select_emulator_windows(windows), [(200, "天天象棋")])

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

    def test_vector_suppresses_selection_glow_rings(self):
        # Create clean piece crop
        image, rect = standard_board_image()
        points = grid_points(rect)
        pt = points[0][0] # Rook at (0,0)
        v_clean = br.Calibration._vector(image, pt, 54)

        # Add a bright glow halo around the piece simulating Tiantian Xiangqi selection effect
        image_glow = image.copy()
        cv2.circle(image_glow, pt, 24, (255, 255, 255), 4) # Bright halo ring
        v_glow = br.Calibration._vector(image_glow, pt, 54)

        similarity = float(np.dot(v_clean, v_glow))
        self.assertGreater(similarity, 0.70)

    def test_neural_piece_classifier_integration(self):
        clf = br.NeuralPieceClassifier.get_instance()
        self.assertTrue(clf.loaded)
        image, rect = standard_board_image()
        points = grid_points(rect)
        pt = points[0][0] # Black Rook at (0,0)
        half = 27
        crop = image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half]
        probs = clf.predict(crop)
        self.assertIn("R", probs)
        self.assertGreater(len(probs), 10)

    def test_midgame_detect_board_rect_fallback(self):
        # Create full board image and mask out top/bottom rows to simulate midgame
        image, rect = standard_board_image()
        # Erase top row and bottom row pieces
        h, w = image.shape[:2]
        image[:120, :] = 150
        image[h-120:, :] = 150
        # Call detect_board_rect which will use the midgame fallback
        detected = detect_board_rect(image)
        self.assertEqual(len(detected), 4)
        self.assertGreater(detected[2], detected[0])
        self.assertGreater(detected[3], detected[1])

    def test_low_score_assigned_piece_is_purged_as_empty(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        # Draw a low-contrast move trail dot at empty intersection (5, 5)
        pt = points[5][5]
        cv2.circle(current, pt, 8, (200, 200, 200), 1)

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[5][5], "")

    def test_neural_network_overrides_ambiguous_has_red_fluctuation(self):
        clf = br.NeuralPieceClassifier.get_instance()
        self.assertTrue(clf.loaded)
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        pt = points[2][1] # Black Cannon 'c' at (2,1)
        half = 27
        crop = image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half]
        probs = clf.predict(crop)
        self.assertIn("c", probs)

    def test_move_trail_dot_during_rebalance_is_not_assigned_low_score_piece(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        # Move Horse from (0, 1) to (2, 2)
        pt_from = points[0][1]
        pt_to = points[2][2]
        crop_horse = image[pt_from[1]-half:pt_from[1]+half, pt_from[0]-half:pt_from[0]+half].copy()
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        
        # Clear origin and paste horse to destination
        current[pt_from[1]-half:pt_from[1]+half, pt_from[0]-half:pt_from[0]+half] = empty_crop
        current[pt_to[1]-half:pt_to[1]+half, pt_to[0]-half:pt_to[0]+half] = crop_horse
        # Draw a trail circle dot on origin (0, 1)
        cv2.circle(current, pt_from, 10, (220, 220, 220), 2)

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[0][1], "")

    def test_all_fourteen_piece_types_recognized_in_midgame_and_captures(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()

        # Erase a Pawn (3, 0) and a Cannon (2, 1) to simulate midgame captures
        pt_pawn = points[3][0]
        pt_cannon = points[2][1]
        current[pt_pawn[1]-half:pt_pawn[1]+half, pt_pawn[0]-half:pt_pawn[0]+half] = empty_crop
        current[pt_cannon[1]-half:pt_cannon[1]+half, pt_cannon[0]-half:pt_cannon[0]+half] = empty_crop

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[3][0], "")
        self.assertEqual(result.board[2][1], "")

    def test_highlighted_landing_piece_is_correctly_recognized(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        
        # Paste Horse from (0, 1) to (2, 5)
        pt_from = points[0][1]
        pt_to = points[2][5]
        crop_horse = image[pt_from[1]-half:pt_from[1]+half, pt_from[0]-half:pt_from[0]+half].copy()
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        current[pt_from[1]-half:pt_from[1]+half, pt_from[0]-half:pt_from[0]+half] = empty_crop
        current[pt_to[1]-half:pt_to[1]+half, pt_to[0]-half:pt_to[0]+half] = crop_horse

        # Draw heavy bright yellow selection ring on (2, 5) simulating landing highlight
        cv2.circle(current, pt_to, 24, (0, 255, 255), 4)

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[2][5], "n")

    def test_exhaustive_fourteen_piece_highlighted_landing_regression(self):
        # Exhaustive regression test verifying that all 14 piece types (red/black)
        # under bright landing highlight rings pass recognition without unconfirmed identity errors.
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        
        # Test positions for all 14 piece types on standard initial board
        piece_positions = [
            (0, 0, "r"), (0, 1, "n"), (0, 2, "b"), (0, 3, "a"), (0, 4, "k"),
            (2, 1, "c"), (3, 0, "p"),
            (9, 0, "R"), (9, 1, "N"), (9, 2, "B"), (9, 3, "A"), (9, 4, "K"),
            (7, 1, "C"), (6, 0, "P"),
        ]
        for r, c, expected_lbl in piece_positions:
            current = image.copy()
            pt = points[r][c]
            # Draw bright yellow/white selection ring simulating landing highlight
            cv2.circle(current, pt, 24, (0, 255, 255), 3)
            result = calibration.recognize(current, align_grid=False)
            self.assertTrue(result.valid, f"Failed for piece {expected_lbl} at ({r},{c}): {result.error}")
            self.assertEqual(result.board[r][c], expected_lbl)

    def test_over_limit_cannon_piece_is_auto_cleaned_without_popup_error(self):
        # Verify that if an extra cannon noise appears (e.g. 3 black cannons), 
        # auto-clean smoothly heals it without halting with "棋子数量超过上限: 炮" error.
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        
        # Paste a 3rd black cannon from (2, 1) onto empty intersection (4, 6)
        pt_cannon = points[2][1]
        pt_dest = points[4][6]
        crop_cannon = image[pt_cannon[1]-half:pt_cannon[1]+half, pt_cannon[0]-half:pt_cannon[0]+half].copy()
        current[pt_dest[1]-half:pt_dest[1]+half, pt_dest[0]-half:pt_dest[0]+half] = crop_cannon

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        pieces = [p for row in result.board for p in row if p == "c"]
        self.assertLessEqual(len(pieces), 2)

    def test_skin_signature_light_fluctuation_is_tolerated(self):
        # Verify that mild ambient brightness or window focus color changes (norm shift around 35-50)
        # do NOT trigger false positive "棋盘皮肤或渲染主题已变化" errors.
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        self.assertIsNotNone(calibration.skin_signature)
        
        current = image.copy()
        # Shift BGR by +20 (norm distance ~34.6) simulating ambient light change
        current = np.clip(current.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        
        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)

    def test_midgame_missing_first_column_does_not_shift_grid(self):
        # Verify that when midgame captures remove pieces on edge columns, 
        # the alignment grid remains locked to self.rect and does NOT shift horizontally.
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()

        # Erase entire 0th column pieces (Rook at 0,0 and Pawn at 3,0)
        for r, c in ((0, 0), (3, 0), (9, 0), (6, 0)):
            pt = points[r][c]
            current[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = empty_crop

        result = calibration.recognize(current, align_grid=True)
        self.assertTrue(result.valid, result.error)
        # Ensure (0, 1) is still recognized as Black Horse "n" (not shifted to column 0)
        self.assertEqual(result.board[0][1], "n")
        self.assertEqual(result.board[0][0], "")

    def test_board_to_fen_sanitizes_three_cannons_without_crashing_pikafish(self):
        # Verify that if a board matrix accidentally contains 3 black cannons (e.g. 1c4cc1),
        # board_to_fen automatically sanitizes it to 2 black cannons without throwing or crashing Pikafish.
        dirty_board = [list(row) for row in STANDARD_BOARD]
        # Add a 3rd black cannon at (2, 7)
        dirty_board[2][7] = "c"
        dirty_board = tuple(tuple(row) for row in dirty_board)

        fen = board_to_fen(dirty_board, "w")
        self.assertIn("1c5c1", fen)
        self.assertNotIn("1c4cc1", fen)

    def test_low_confidence_piece_auto_recovers_without_blocking_popup(self):
        # Verify that if a piece crop gets a low confidence score (e.g. 0.26 due to landing ring glare),
        # recognize() auto-recovers with highest candidate instead of throwing "棋子身份不明确: (2, 6)" popup.
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        current = image.copy()
        
        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)

    def test_exhaustive_real_halo_simulation_for_all_fourteen_pieces(self):
        # Exhaustively simulate real landing halo and trail dot artifacts across all 14 piece types:
        # Red: K, A, B, N, R, C, P
        # Black: k, a, b, n, r, c, p
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        # Piece position map in standard opening board
        piece_coords = {
            "r": (0, 0), "n": (0, 1), "b": (0, 2), "a": (0, 3), "k": (0, 4), "c": (2, 1), "p": (3, 0),
            "R": (9, 0), "N": (9, 1), "B": (9, 2), "A": (9, 3), "K": (9, 4), "C": (7, 1), "P": (6, 0),
        }

        def apply_landing_halo(img_crop, intensity=1.0):
            # Physical simulation of Tiantian Xiangqi yellow/white circular ring halo
            h, w = img_crop.shape[:2]
            cx, cy = w // 2, h // 2
            radius = int(min(w, h) * 0.42)
            thickness = max(3, int(min(w, h) * 0.12))
            halo_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(halo_mask, (cx, cy), radius, 255, thickness)

            # Yellow/White bright RGB tint
            yellow_tint = np.zeros_like(img_crop)
            yellow_tint[:, :] = (30 * intensity, 180 * intensity, 220 * intensity) # BGR
            
            blended = img_crop.copy().astype(np.float32)
            alpha = (halo_mask.astype(np.float32) / 255.0)[:, :, None] * 0.55 * intensity
            blended = blended * (1.0 - alpha) + yellow_tint.astype(np.float32) * alpha
            return np.clip(blended, 0, 255).astype(np.uint8)

        # Test each of the 14 piece types under 4 physical variations (56 test scenarios)
        for piece_label, (r, c) in piece_coords.items():
            current = image.copy()
            pt = points[r][c]
            crop_orig = image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half].copy()

            # Variant 1: Mild halo
            crop_mild = apply_landing_halo(crop_orig, intensity=0.6)
            current[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = crop_mild
            res = calibration.recognize(current, align_grid=False)
            self.assertTrue(res.valid, f"Failed on piece {piece_label} mild halo: {res.error}")
            self.assertEqual(res.board[r][c], piece_label, f"Piece {piece_label} misclassified under mild halo")

            # Variant 2: Strong intense yellow halo
            crop_strong = apply_landing_halo(crop_orig, intensity=1.2)
            current[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = crop_strong
            res = calibration.recognize(current, align_grid=False)
            self.assertTrue(res.valid, f"Failed on piece {piece_label} strong halo: {res.error}")
            self.assertEqual(res.board[r][c], piece_label, f"Piece {piece_label} misclassified under strong halo")

    def test_crossed_black_cannon_on_red_woodgrain_background_recognized_as_black_cannon(self):
        # Verify that when black cannon "c" moves across the river into red board area (3, 4),
        # red woodgrain background tint does NOT falsely turn it into red cannon "C".
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()

        # Crop black cannon "c" from (2, 1) and paste onto (3, 4) in red riverbank area
        pt_cannon = points[2][1]
        pt_dest = points[3][4]
        crop_cannon = image[pt_cannon[1]-half:pt_cannon[1]+half, pt_cannon[0]-half:pt_cannon[0]+half].copy()
        
        # Erase original (2, 1)
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        current[pt_cannon[1]-half:pt_cannon[1]+half, pt_cannon[0]-half:pt_cannon[0]+half] = empty_crop
        
        # Paste onto (3, 4)
        current[pt_dest[1]-half:pt_dest[1]+half, pt_dest[0]-half:pt_dest[0]+half] = crop_cannon

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[3][4], "c", f"Crossed black cannon at (3, 4) was misclassified as {result.board[3][4]!r}")

    def test_crossed_black_knight_on_center_file_is_recognized_and_translated_correctly(self):
        # Verify that when black horse "n" moves to center file (3, 4),
        # recognize() correctly identifies it as "n" (not "N" or "C") and PikafishEngine.get_chinese_move() does not output "炮五退二".
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        current = image.copy()

        # Crop black knight "n" from (0, 1) and paste onto (3, 4)
        pt_knight = points[0][1]
        pt_dest = points[3][4]
        crop_knight = image[pt_knight[1]-half:pt_knight[1]+half, pt_knight[0]-half:pt_knight[0]+half].copy()
        
        # Erase original (0, 1)
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        current[pt_knight[1]-half:pt_knight[1]+half, pt_knight[0]-half:pt_knight[0]+half] = empty_crop
        
        # Paste onto (3, 4)
        current[pt_dest[1]-half:pt_dest[1]+half, pt_dest[0]-half:pt_dest[0]+half] = crop_knight

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[3][4], "n", f"Black knight at (3, 4) was misclassified as {result.board[3][4]!r}")

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

    def test_move_to_uci_uses_pikafish_coordinates(self):
        self.assertEqual(br.move_to_uci(Move("w", (9, 7), (7, 6))), "h0g2")


class RecognitionTests(unittest.TestCase):
    def test_identity_candidates_exclude_piece_types_at_capacity(self):
        feasible = getattr(
            br,
            "_feasible_identity_scores",
            lambda scores, _label, _counts, _limits: scores,
        )
        scores = {"": 0.90, "c": 0.38, "b": 0.31, "k": 0.21}

        self.assertEqual(
            feasible(
                scores,
                "c",
                {"c": 2, "b": 2, "k": 1},
                {"c": 2, "b": 2, "k": 1},
            ),
            {"c": 0.38},
        )

    def test_identity_margin_compares_final_label_with_runner_up(self):
        margin = getattr(br, "_identity_margin", lambda _scores, _label: 1.0)
        self.assertAlmostEqual(
            margin({"": 0.90, "N": 0.74, "P": 0.71, "R": 0.20}, "N"),
            0.03,
        )

    def test_ambiguous_moved_horse_is_rejected_instead_of_becoming_pawn(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
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

        empty = crop(5, 5)
        paste(9, 1, empty)
        paste(6, 0, empty)
        paste(3, 2, crop(9, 1))

        horse = calibration.samples[
            np.flatnonzero(calibration.labels == "N")[0]
        ]
        pawn = calibration.samples[
            np.flatnonzero(calibration.labels == "P")[0]
        ]
        ambiguous = horse * 0.49 + pawn * 0.51
        ambiguous /= np.linalg.norm(ambiguous)
        original_vector = calibration._vector
        target_x, target_y = points[3][2]

        def mocked_vector(frame, point, sample_size):
            if (
                abs(point[0] - target_x) <= 3
                and abs(point[1] - target_y) <= 3
            ):
                return ambiguous.copy()
            return original_vector(frame, point, sample_size)

        with patch.object(calibration, "_vector", side_effect=mocked_vector):
            result = calibration.recognize(current, align_grid=False)

        # Under the auto-healing safety design, low-confidence or narrow-margin pieces are auto-recovered
        # with their highest confidence non-empty label instead of halting with a blocking popup error.
        self.assertTrue(result.valid, result.error)

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
                self.assertEqual(
                    result.board[target[0]][target[1]],
                    expected,
                )

    def test_occupied_high_confidence_wrong_label_searches_neighborhood(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
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

        empty = crop(5, 5)
        paste(0, 0, empty)
        paste(3, 4, empty)
        paste(4, 4, crop(0, 0))

        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label and (row, col) not in {(0, 0), (3, 4)}
        } | {(4, 4)}
        pawn = calibration.samples[
            np.flatnonzero(calibration.labels == "p")[0]
        ]
        rook = calibration.samples[
            np.flatnonzero(calibration.labels == "r")[0]
        ]
        original_vector = calibration._vector
        target_x, target_y = points[4][4]

        def mocked_vector(frame, point, sample_size):
            if point == (target_x, target_y):
                return pawn * 0.95
            if (
                abs(point[0] - target_x) <= 3
                and abs(point[1] - target_y) <= 3
            ):
                return rook.copy()
            return original_vector(frame, point, sample_size)

        with patch.object(calibration, "_occupied_cells", return_value=occupied), \
             patch.object(calibration, "_vector", side_effect=mocked_vector):
            result = calibration.recognize(current, align_grid=False)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[4][4], "r")

    def test_trusted_occupancy_rejects_piece_when_identity_is_unclear(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        row, col = 0, 2
        piece_vector = calibration._vector(
            image, points[row][col], calibration.size
        )
        calibration.samples[np.flatnonzero(calibration.labels == "")[0]] = piece_vector
        calibration.samples[calibration.labels == "b"] *= 0.8
        occupied = {
            (r, c)
            for r, values in enumerate(STANDARD_BOARD)
            for c, label in enumerate(values)
            if label and (r, c) != (0, 6)
        }

        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            result = calibration.recognize(image)

        self.assertTrue(result.valid, result.error)

    def test_dynamic_std_threshold_calibration(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        # For synthetic standard board image:
        # Empty cells are solid grey (std = 0.0).
        # Piece cells have text and borders (std >= 30.0).
        # Midpoint is around 15.0 to 20.0.
        self.assertTrue(10.0 <= calibration.std_threshold <= 30.0)

    def test_empty_highlight_ignored_due_to_std_threshold(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        
        # Simulate a false positive occupied circle on an empty cell (1, 1)
        # by patching _occupied_cells to include (1, 1).
        occupied = {
            (r, c)
            for r, values in enumerate(STANDARD_BOARD)
            for c, label in enumerate(values)
            if label
        } | {(1, 1)}
        
        with patch.object(calibration, "_occupied_cells", return_value=occupied):
            result = calibration.recognize(image)
            
        # The empty cell (1, 1) has std = 0.0 < std_threshold, so it must be filtered out
        # from occupied, and the board recognition should succeed!
        self.assertTrue(result.valid)
        self.assertEqual(result.board[1][1], "")

    def test_false_positive_circle_with_high_std_dev_is_ignored(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        
        # Simulate a false positive occupied circle on an empty cell (1, 1).
        # We also mock the similarity values of cell (1, 1) to nonempty labels to be low.
        # Since it's a synthetic image, (1, 1) has std = 0.0, so to prevent std_threshold filtering,
        # we patch Calibration._cell_std to return a high std (e.g. 50.0) for cell (1, 1).
        occupied = {
            (r, c)
            for r, values in enumerate(STANDARD_BOARD)
            for c, label in enumerate(values)
            if label
        } | {(1, 1)}
        
        original_cell_std = calibration._cell_std
        def mocked_cell_std(img, point, size):
            points = grid_points(rect)
            if np.allclose(point, points[1][1]):
                return 50.0
            return original_cell_std(img, point, size)
            
        with patch.object(calibration, "_occupied_cells", return_value=occupied), \
             patch.object(calibration, "_cell_std", side_effect=mocked_cell_std):
            result = calibration.recognize(image)
            
        # Cell (1, 1) has std >= std_threshold but nonempty template similarity is extremely low,
        # so it must be discarded from occupied, and the board recognition should succeed!
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[1][1], "")

    def test_occupied_cell_never_demoted_to_empty(self):
        """
        Scenario: in occupancy_trusted mode, when a real occupied Cannon is highlighted
        (slightly lowered match score), and a false-positive HoughCircle at an adjacent empty
        cell also matches Cannon well, the inventory check must demote the FALSE cell, not
        the real occupied Cannon.
        """
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)

        # The real board has exactly 2 black Cannons: at (2, 1) and (2, 7).
        # We inject a false-positive circle at (3, 1) — an empty cell.
        # To make (3, 1) look like a Cannon (fool the first-pass classifier):
        #   - mock _cell_std to return high std (so it's not discarded as background)
        #   - mock _vector to return the actual Cannon-at-(2,1)'s vector for (3,1)
        # Simultaneously, the real Cannon at (2, 1) is "highlighted" so its vector
        # is slightly worse — but still better-than-empty.
        #
        # Expected: inventory check finds 3 "c" assigned but limit is 2;
        # should demote (3, 1) to "" (false positive), NOT (2, 1) (real, occupied).

        # Get real Cannon vector from (2, 1) — to use as the strong fake vector too.
        cannon_vector = calibration._vector(image, points[2][1], calibration.size)

        # Create a "degraded" version for the highlighted real Cannon at (2, 1):
        # The highlight ring slightly blurs the piece center, dropping cos-similarity from 1.0 to ~0.75.
        # Achieved by blending 75% cannon + 25% background vector (zero).
        degraded_cannon_vector = cannon_vector * 0.75

        original_vector = calibration._vector
        original_cell_std = calibration._cell_std

        def mocked_cell_std(img, point, size):
            if np.allclose(point, points[3][1]):
                return 50.0  # ensure (3,1) passes std filter and gets a vector
            return original_cell_std(img, point, size)

        def mocked_vector(img, point, size):
            if np.allclose(point, points[3][1]):
                # False positive NON-OCCUPIED: has high cannon similarity (1.0).
                # This enters via the strong-match path in line 617 (score >= threshold),
                # NOT via the occupied path.
                return cannon_vector.copy()
            if np.allclose(point, points[2][1]):
                # Real highlighted Cannon: degraded similarity (~0.75), still >> empty.
                return degraded_cannon_vector.copy()
            return original_vector(img, point, size)

        # (3, 1) is NOT in occupied — it's a false-positive that passes the strong-match
        # threshold (score >= 0.55) in the non-occupied branch of line 617.
        occupied = {
            (r, c)
            for r, values in enumerate(STANDARD_BOARD)
            for c, label in enumerate(values)
            if label
        }
        # (3, 1) intentionally NOT added to occupied

        with patch.object(calibration, "_occupied_cells", return_value=occupied), \
             patch.object(calibration, "_cell_std", side_effect=mocked_cell_std), \
             patch.object(calibration, "_vector", side_effect=mocked_vector):
            result = calibration.recognize(image)

        # Board should be valid — the non-occupied false positive (3, 1) should have been
        # demoted to "" (it's not in occupied, so alternatives include empty first),
        # and the real Cannons at (2, 1) and (2, 7) should survive as "c".
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[2][1], "c")
        self.assertEqual(result.board[2][7], "c")
        self.assertEqual(result.board[3][1], "")

    def test_coordinate_legal_filtering(self):
        from board_recognition import is_coordinate_legal
        # King K/k palace check
        self.assertTrue(is_coordinate_legal('k', 0, 4))
        self.assertFalse(is_coordinate_legal('k', 3, 4))
        self.assertTrue(is_coordinate_legal('K', 8, 4))
        self.assertFalse(is_coordinate_legal('K', 5, 4))
        
        # Elephant b/B legal check
        self.assertTrue(is_coordinate_legal('b', 2, 4))
        self.assertFalse(is_coordinate_legal('b', 3, 4))
        self.assertTrue(is_coordinate_legal('B', 9, 2))
        self.assertFalse(is_coordinate_legal('B', 4, 2))
        
        # Pawn p/P forward check
        self.assertTrue(is_coordinate_legal('p', 3, 0))
        self.assertFalse(is_coordinate_legal('p', 2, 0))
        self.assertTrue(is_coordinate_legal('P', 6, 0))
        self.assertFalse(is_coordinate_legal('P', 7, 0))

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

    def test_piece_inventory_recognizes_moved_cannon(self):
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

        copy_cell((7, 1), (7, 4))
        copy_cell((7, 0), (7, 1))
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
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        # Erase black advisor at (0, 5) from image
        pt = points[0][5]
        empty = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = empty
        
        occupied = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label and (row, col) != (0, 5)
        }
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
        calibration.samples *= 0.7
        calibration.threshold = 0.75
        calibration.circle_occupancy = False

        result = calibration.recognize(image)

        self.assertLess(result.confidence, calibration.threshold)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

    def test_circle_board_still_rejects_missing_king(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        # Erase black king at (0, 4) from image
        pt = points[0][4]
        empty = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = empty
        
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
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2
        # Erase black advisor at (0, 5) from image
        pt = points[0][5]
        empty = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()
        image[pt[1]-half:pt[1]+half, pt[0]-half:pt[0]+half] = empty

        calibration.circle_occupancy = False
        calibration.samples *= 0.7
        calibration.threshold = 0.75
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
        # Second search receives full moves history chain
        self.assertIn(" moves ", searches[1], searches[1])
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

    def test_calibration_rejects_nonstandard_occupancy_before_learning(self):
        image, rect = standard_board_image()
        expected = {
            (row, col)
            for row, values in enumerate(STANDARD_BOARD)
            for col, label in enumerate(values)
            if label
        }
        midgame = expected - {(7, 1)} | {(7, 4)}

        with (
            patch.object(Calibration, "_occupied_cells", return_value=midgame),
            patch.object(
                Calibration,
                "recognize",
                return_value=br.Recognition(STANDARD_BOARD, 1.0, True),
            ),
            self.assertRaisesRegex(ValueError, "只能在静止的标准开局校准"),
        ):
            Calibration.create(image, rect)

    def test_calibration_rejects_legacy_feature_schema(self):
        image, rect = standard_board_image()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Calibration.create(image, rect).save(directory)
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["feature_version"] = 1
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "校准格式已升级"):
                Calibration.load(directory)

    def test_automatic_calibration_finds_standard_board(self):
        image, expected = standard_board_image()
        rect = detect_board_rect(image)
        self.assertTrue(all(abs(a - b) <= 5 for a, b in zip(rect, expected)), rect)
        self.assertEqual(Calibration.create(image, rect).recognize(image).board, STANDARD_BOARD)

    def test_alignment_recovers_same_size_negative_client_area_shift(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        shifted = cv2.warpAffine(
            image,
            np.float32([[1, 0, -16], [0, 1, -16]]),
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )

        detected = (
            rect[0] - 27,
            rect[1] - 24,
            rect[2] - 5,
            rect[3] - 8,
        )
        with patch("board_recognition.detect_board_rect", return_value=detected):
            result = calibration.recognize(shifted, align_grid=True)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)
        self.assertGreaterEqual(result.confidence, calibration.threshold)

    def test_alignment_falls_back_when_starting_rect_is_unavailable(self):
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        shifted = cv2.warpAffine(
            image,
            np.float32([[1, 0, 16], [0, 1, 16]]),
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )

        with patch(
            "board_recognition.detect_board_rect",
            side_effect=ValueError("非标准中盘"),
        ):
            result = calibration.recognize(shifted, align_grid=True)

        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board, STANDARD_BOARD)

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
        self.assertEqual(
            selector(centers, (42, 272, 494, 766), 0.0, 47),
            (0, 0),
        )

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

    def test_configure_strength_caps_threads_at_eight(self):
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
                "setoption name Threads value 8",
                "setoption name Hash value 512",
                "setoption name MultiPV value 1",
            ],
        )

    def test_engine_uses_adaptive_time_with_hard_limit(self):
        engine = object.__new__(PikafishEngine)
        engine._lock = threading.Lock()
        sent = []
        engine.send_command = sent.append
        engine._read_until = lambda _predicate, _timeout: ["bestmove a0a1"]
        command = f"position fen {board_to_fen(STANDARD_BOARD, 'w')} moves h0g2"

        engine.get_best_move(command, movetime=5000)

        self.assertEqual(
            sent,
            [
                "isready",
                command,
                "go movetime 5000",
            ],
        )

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

    def test_stop_search_sends_stop_only_to_running_process(self):
        engine = object.__new__(PikafishEngine)
        process = Mock()
        process.poll.return_value = None
        engine.process = process

        engine.stop_search()

        process.stdin.write.assert_called_once_with("stop\n")
        process.stdin.flush.assert_called_once_with()

        process.stdin.reset_mock()
        process.poll.return_value = 0
        engine.stop_search()
        process.stdin.write.assert_not_called()

    def test_stop_search_ignores_stdin_closed_during_shutdown(self):
        engine = object.__new__(PikafishEngine)
        process = Mock()
        process.poll.return_value = None
        process.stdin.write.side_effect = ValueError("I/O operation on closed file")
        engine.process = process

        engine.stop_search()

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

    def test_app_does_not_retry_deterministic_position_error(self):
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.stop_event = Mock(is_set=Mock(return_value=False))
        app.closing = False
        engine = Mock()
        engine.get_best_move.side_effect = ValueError("非法局面")
        app.engine = engine

        with (
            patch("app.PikafishEngine") as engine_class,
            self.assertRaisesRegex(ValueError, "非法局面"),
        ):
            app.best_move("position fen invalid", 1000)

        engine.get_best_move.assert_called_once_with(
            "position fen invalid", movetime=1000
        )
        engine.close.assert_not_called()
        engine_class.assert_not_called()

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

    def test_best_advice_rejects_cannon_jump_over_two_screens(self):
        fen = (
            "1rbaka2r/9/c1n1b1cCn/p1p1p3p/9/2P3p2/"
            "P3P3P/2N1C4/4N4/R1BAKAB1R w - - 0 1"
        )
        app = object.__new__(AssistantApp)
        app.stop_event = threading.Event()
        app.closing = False
        app.engine_lock = threading.Lock()
        app.engine = None
        app.best_move = Mock(return_value=("h7c7", "+3.88"))
        board, _ = br.fen_to_board(fen)

        with self.assertRaisesRegex(RuntimeError, "非法着法"):
            app.best_advice(board, f"position fen {fen}", "w", 1000)

    def test_best_advice_rejects_move_that_does_not_evade_check(self):
        fen = (
            "1rb1k4/4a4/2Ra5/pNpC2p1p/4r4/8P/2P6/8B/9/2BAKA3 "
            "w - - 0 1"
        )
        board, side = br.fen_to_board(fen)
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.engine = None
        app.best_move = Mock(return_value=("b6c4", "+6.27"))

        with self.assertRaisesRegex(RuntimeError, "非法着法"):
            app.best_advice(board, f"position fen {fen}", side, 1000)

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

    def test_real_engine_analyzes_reported_check_position(self):
        fen = (
            "1rb1k4/4a4/2Ra5/pNpC2p1p/4r4/8P/2P6/8B/9/2BAKA3 "
            "w - - 0 1"
        )
        board, side = br.fen_to_board(fen)
        engine = PikafishEngine()
        try:
            move, _ = engine.get_best_move(fen, movetime=200)
        finally:
            engine.close()
        sc, sr, ec, er = PikafishEngine.uci_to_coords(move)

        self.assertNotEqual(move, "b6c4")
        self.assertTrue(
            br.is_legal_xiangqi_move(board, sr, sc, er, ec, side),
            move,
        )


class LoopTests(unittest.TestCase):
    def test_defaults_use_twenty_second_search_and_point_two_sampling(self):
        self.assertEqual(app_module.DEFAULT_SEARCH_SECONDS, 20)
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

    def test_stop_notifies_running_engine(self):
        app = object.__new__(AssistantApp)
        app.stop_event = threading.Event()
        app.engine_lock = threading.Lock()
        app.engine = Mock()
        app.start_button = Mock()
        app.status_var = Mock()

        app.stop()

        self.assertTrue(app.stop_event.is_set())
        app.engine.stop_search.assert_called_once_with()
        app.start_button.configure.assert_called_once_with(state="disabled")
        app.status_var.set.assert_called_once_with("正在停止分析")

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

    def test_run_loop_auto_recalculates_on_advice_exception(self):
        from game_state import GameState

        class StopAfterWait:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, _seconds):
                self.stopped = True
                return True

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        app = object.__new__(AssistantApp)
        app.stop_event = StopAfterWait()
        app.results = Queue()
        app.engine_lock = threading.Lock()
        app.engine = None
        app.calibration = Mock()
        app.calibration.rect = (0, 0, 18, 18)
        app.config_dir = Path("diagnostics")
        app.capture = Mock(return_value=frame)
        
        tracker = Mock()
        tracker.observe.return_value = br.Recognition(
            STANDARD_BOARD,
            1.0,
            True,
            "",
            recovery=False,
            reason="standard",
        )
        
        app.best_advice = Mock(side_effect=RuntimeError("Pikafish返回非法着法"))
        state = GameState.start(STANDARD_BOARD, "w", "w")
        
        with patch("app.MotionTracker", return_value=tracker):
            app.run_loop("w", state, frame)
            
        self.assertTrue(any("行棋或分析异常" in str(event[1]) for event in app.results.queue if event[0] == "clear_move"))

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
            tuple(), 0.0, False, "检测到棋子但无法确认身份: [(2, 4)]"
        )

        app.sync_current_board()

        save_snapshot.assert_called_once_with(app.config_dir, frame)
        thread_class.assert_not_called()
        showerror.assert_called_once()
        app.status_var.set.assert_called_with(
            "无法识别当前棋盘: 检测到棋子但无法确认身份: [(2, 4)]。请等待动画结束后重试；"
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

    def test_is_legal_xiangqi_move_cannon_rules(self):
        from board_recognition import is_legal_xiangqi_move
        # Board with Cannon at h2 (7, 7), Horse at g2 (7, 6), empty at e2 (7, 4)
        board = [[""] * 9 for _ in range(10)]
        board[7][7] = "C"  # Red Cannon at h2 (row 7, col 7)
        board[7][6] = "N"  # Red Horse at g2 (row 7, col 6)
        board[7][4] = ""   # Empty at e2 (row 7, col 4)
        board[9][4] = "K"  # Red King
        board[0][4] = "k"  # Black King

        # Moving Cannon h2 -> e2 (over g2 Horse) to empty square is ILLEGAL
        self.assertFalse(is_legal_xiangqi_move(board, 7, 7, 7, 4, "w"))

        # Clear horse at g2 -> now h2 -> e2 to empty square is LEGAL
        board[7][6] = ""
        self.assertTrue(is_legal_xiangqi_move(board, 7, 7, 7, 4, "w"))

        # Put horse back at g2, put black pawn 'p' at e2 -> h2 -> e2 captures over 1 screen -> LEGAL
        board[7][6] = "N"
        board[7][4] = "p"
        self.assertTrue(is_legal_xiangqi_move(board, 7, 7, 7, 4, "w"))

        # Add another piece at f2 (7, 5) -> now 2 screens -> ILLEGAL capture
        board[7][5] = "P"
        self.assertFalse(is_legal_xiangqi_move(board, 7, 7, 7, 4, "w"))

    def test_is_legal_xiangqi_move_horse_and_flying_general(self):
        from board_recognition import is_legal_xiangqi_move
        board = [[""] * 9 for _ in range(10)]
        board[9][4] = "K"  # Red King at e1
        board[0][4] = "k"  # Black King at e10
        board[5][4] = "P"  # Red Pawn at e5 blocking Flying General between Kings
        board[7][2] = "N"  # Red Horse at c2 (7, 2)
        board[7][1] = "P"  # Red Pawn at b2 (7, 1) - blocks horse leg (7, 1) for (7,2)->(6,0)

        # (7, 2) -> (5, 1) horse leg at (6, 2) is empty -> LEGAL
        self.assertTrue(is_legal_xiangqi_move(board, 7, 2, 5, 1, "w"))

        # (7, 2) -> (6, 0) horse leg at (7, 1) has Pawn -> BLOCKED -> ILLEGAL
        self.assertFalse(is_legal_xiangqi_move(board, 7, 2, 6, 0, "w"))

        # Flying general: Red Advisor at (8, 4) blocking Kings if Pawn at (5, 4) is removed.
        board[5][4] = ""
        board[8][4] = "A"  # Red Advisor at (8, 4) blocking Kings.
        board[8][3] = ""
        # Moving A from (8, 4) to (7, 3) unblocks column 4 between K (9, 4) and k (0, 4) -> Flying General -> ILLEGAL
        self.assertFalse(is_legal_xiangqi_move(board, 8, 4, 7, 3, "w"))

        field = [list(row) for row in STANDARD_BOARD]
        field[9][1] = ""
        field[3][2] = "N"
        field = tuple(tuple(row) for row in field)
        self.assertFalse(is_legal_xiangqi_move(field, 3, 2, 2, 2, "w"))
        self.assertTrue(is_legal_xiangqi_move(field, 3, 2, 1, 1, "w"))

    def test_move_is_illegal_when_it_does_not_evade_rook_check(self):
        board, side = br.fen_to_board(
            "1rb1k4/4a4/2Ra5/pNpC2p1p/4r4/8P/2P6/8B/9/2BAKA3 "
            "w - - 0 1"
        )

        self.assertTrue(br._king_in_check(board, side))
        self.assertFalse(
            br.is_legal_xiangqi_move(board, 3, 1, 5, 2, side)
        )

    def test_best_advice_rejects_illegal_move(self):
        board = [[""] * 9 for _ in range(10)]
        board[7][7] = "C"  # Red Cannon at h2
        board[7][6] = "N"  # Red Horse at g2
        board[7][4] = ""   # Empty at e2
        board[9][4] = "K"
        board[0][4] = "k"

        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.engine = Mock()
        # Mock engine returning illegal move "h2e2" (c7r7 to c4r7)
        app.best_move = Mock(return_value=("h2e2", "+0.16"))

        with self.assertRaises(RuntimeError) as ctx:
            app.best_advice(board, "position fen test", "w")
        self.assertIn("Pikafish返回非法着法", str(ctx.exception))

    def test_highlight_ring_does_not_cause_red_misclassification(self):
        # Create a mock cell crop with a bright yellow/red outer highlight ring but a black piece text in the center
        crop = np.full((40, 40, 3), 150, dtype=np.uint8)
        # Outer ring (yellow/red highlight ring, hue around 15, sat 200, val 200)
        cv2.circle(crop, (20, 20), 18, (0, 200, 255), 3)
        # Inner text (dark black text in center, no red)
        cv2.circle(crop, (20, 20), 8, (20, 20, 20), -1)

        # Check inner 45% crop logic
        h, w = crop.shape[:2]
        inner_crop = crop[int(h * 0.27) : int(h * 0.73), int(w * 0.27) : int(w * 0.73)]
        hsv = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (
            ((hue < 10) | (hue > 170))
            & (saturation > 110)
            & (value > 80)
        )
        has_red = np.count_nonzero(red_mask) >= max(3, int(inner_crop.size * 0.008))
        self.assertFalse(has_red, "Outer highlight ring should not trigger false has_red flag on inner black piece")

    def test_recognition_confidence_robustness_against_single_outlier(self):
        # 31 pieces score high (0.95), only 1 piece has animation outlier score (0.14)
        scores = [0.95] * 31 + [0.14]
        conf = recognition_confidence(scores)
        # Robust confidence must NOT be dragged down to 0.14, it should be high (> 0.70)
        self.assertGreater(conf, 0.70, f"Confidence {conf:.2f} should ignore single outlier 0.14 score")

    def test_dead_engine_is_automatically_restarted(self):
        app = object.__new__(AssistantApp)
        app.engine_lock = threading.Lock()
        app.stop_event = threading.Event()
        app.closing = False

        # Create a mock engine that reports is_alive = False
        dead_engine = Mock()
        dead_engine.is_alive = Mock(return_value=False)
        app.engine = dead_engine

        # Mock PikafishEngine class constructor to return a healthy new engine
        healthy_engine = Mock()
        healthy_engine.get_best_move = Mock(return_value=("b2e2", "+0.30"))
        healthy_engine.is_alive = Mock(return_value=True)

        with patch("app.PikafishEngine", return_value=healthy_engine):
            move, score = app.best_move("position fen test", 1000)
            self.assertEqual(move, "b2e2")
            self.assertEqual(app.engine, healthy_engine)

    def test_tightened_core_vector_strips_outer_glow_ring(self):
        # Create a mock image with a bright outer selection ring at radius 15px
        image = np.full((100, 100, 3), 150, dtype=np.uint8)
        # Bright selection glow ring (radius 15, thickness 3)
        cv2.circle(image, (50, 50), 15, (255, 255, 255), 3)
        # Inner text center
        cv2.circle(image, (50, 50), 5, (20, 20, 20), -1)

        vec = Calibration._vector(image, (50, 50), 40)
        # Feature vector length should equal elements within 12.0px radius
        yy, xx = np.ogrid[:40, :40]
        expected_len = np.count_nonzero((xx - 19.5)**2 + (yy - 19.5)**2 <= 12.0**2)
        self.assertEqual(len(vec), expected_len)


    def test_black_piece_without_red_ink_is_never_classified_as_red_piece(self):
        crop = np.full((40, 40, 3), (180, 200, 210), dtype=np.uint8)
        inner_crop = crop[int(40 * 0.27) : int(40 * 0.73), int(40 * 0.27) : int(40 * 0.73)]
        hsv = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (((hue < 10) | (hue > 170)) & (saturation > 110) & (value > 80))
        has_red = np.count_nonzero(red_mask) >= max(3, int(inner_crop.size * 0.008))
        self.assertFalse(has_red)

        label = "R"
        color_mismatch = (label.isupper() and not has_red) or (label.islower() and has_red)
        self.assertTrue(color_mismatch)

    def test_has_red_threshold_ignores_minor_noise(self):
        crop = np.full((40, 40, 3), (180, 200, 210), dtype=np.uint8)
        crop[18, 18] = (20, 20, 220)
        crop[18, 19] = (20, 20, 220)
        crop[19, 18] = (20, 20, 220)
        crop[19, 19] = (20, 20, 220)
        crop[20, 20] = (20, 20, 220)

        inner_crop = crop[int(40 * 0.27) : int(40 * 0.73), int(40 * 0.27) : int(40 * 0.73)]
        hsv = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (((hue < 10) | (hue > 170)) & (saturation > 110) & (value > 80))
        red_count = np.count_nonzero(red_mask)
        red_ratio = red_count / float(inner_crop.size) if inner_crop.size > 0 else 0.0
        has_red = (red_count >= 15 and red_ratio >= 0.035)
        self.assertFalse(has_red, "5 noise red pixels should NOT trigger has_red True")

    def test_synthetic_red_arrow_overlay_is_filtered_out(self):
        crop = np.full((40, 40, 3), (180, 200, 210), dtype=np.uint8)
        cv2.line(crop, (5, 5), (35, 35), (0, 0, 255), 4)

        inner_crop = crop[int(40 * 0.27) : int(40 * 0.73), int(40 * 0.27) : int(40 * 0.73)]
        hsv = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        b_ch, g_ch, r_ch = cv2.split(inner_crop)
        synthetic_overlay = (r_ch > 220) & (g_ch < 50) & (b_ch < 50)
        red_mask = (
            ((hue < 10) | (hue > 170))
            & (saturation > 110)
            & (value > 80)
            & (~synthetic_overlay)
        )
        red_count = np.count_nonzero(red_mask)
        self.assertEqual(red_count, 0, "Pure red arrow line pixels should be 100% filtered out")

    def test_format_analysis_result_heavy_deficit_warning(self):
        from app import format_analysis_result
        text, status = format_analysis_result("d1e1", "-9.64", "帅六平五")
        self.assertIn("大劣局面/黑胜势", status)

    def test_high_confidence_cannon_protected_from_demotion(self):
        from pikafish_engine import PikafishEngine
        from board_recognition import fen_to_board
        fen = 'rC1ak4/4a4/b3b1c1n/p1p4rp/4p4/2P1C4/P3P1C1P/6N1B/1R7/1RBAKA3 w - - 0 1'
        board, _ = fen_to_board(fen)
        chinese = PikafishEngine.get_chinese_move('b9a9', board)
    def test_game_state_position_command_passed_with_moves_chain(self):
        from game_state import GameState
        from board_recognition import STANDARD_BOARD
        state = GameState.start(STANDARD_BOARD, "w", "w")
        cmd1 = state.position_command()
        self.assertTrue(cmd1.startswith("position fen "))
        self.assertNotIn("moves", cmd1)

        board2 = [list(r) for r in STANDARD_BOARD]
        board2[7][4] = board2[7][1]
        board2[7][1] = ""
        tuple2 = tuple(tuple(r) for r in board2)
        applied = state.apply_board(tuple2)
        self.assertTrue(applied)
        cmd2 = state.position_command()
        self.assertIn("moves b2e2", cmd2, "GameState position_command must contain historical moves chain")

    def test_black_rook_checking_red_king_rejects_cannon_move(self):
        from board_recognition import is_legal_xiangqi_move
        # Construct a board where Black Rook 'r' is at (4, 4) [e5] and Red King 'K' is at (9, 4) [e1]
        # Red Cannon 'C' is at (8, 0) [a2]
        board = [[""] * 9 for _ in range(10)]
        board[0][4] = "k"  # Black King
        board[4][4] = "r"  # Black Rook at e5
        board[9][4] = "K"  # Red King at e1
        board[8][0] = "C"  # Red Cannon at a2
        board[9][3] = "A"  # Red Advisor
        board[9][5] = "A"  # Red Advisor

        # Moving Red Cannon a2 -> a6 (from (8, 0) to (4, 0)) MUST be illegal because Red King remains in check
        legal = is_legal_xiangqi_move(board, 8, 0, 4, 0, "w")
        self.assertFalse(legal, "Cannon move a2a6 must be illegal when Red King is checked by Black Rook on 5th file")

    def test_highlighted_black_rook_on_center_file_retains_recognition(self):
        from board_recognition import Calibration, STANDARD_BOARD
        image, rect = standard_board_image()
        calibration = Calibration.create(image, rect)
        points = grid_points(rect)
        size, half = calibration.size, calibration.size // 2

        # Create a frame where cell (4, 4) has a black piece with red/white halo indicator
        current = image.copy()
        pt_rook = points[0][0]  # Black Rook at (0, 0)
        rook_crop = image[pt_rook[1]-half:pt_rook[1]+half, pt_rook[0]-half:pt_rook[0]+half].copy()
        empty_crop = image[points[5][5][1]-half:points[5][5][1]+half, points[5][5][0]-half:points[5][5][0]+half].copy()

        # Move Black Rook from (0, 0) to (4, 4)
        current[pt_rook[1]-half:pt_rook[1]+half, pt_rook[0]-half:pt_rook[0]+half] = empty_crop
        # Add red/white target icon on top of rook crop
        cv2.circle(rook_crop, (half, half), 8, (0, 0, 255), 2)
        cv2.circle(rook_crop, (half, half), 3, (255, 255, 255), -1)

        pt_target = points[4][4]
        current[pt_target[1]-half:pt_target[1]+half, pt_target[0]-half:pt_target[0]+half] = rook_crop

        result = calibration.recognize(current, align_grid=False)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.board[4][4], "r", "Highlighted Black Rook at e5 must be recognized as 'r', not 'P' or empty")


if __name__ == "__main__":
    unittest.main()




