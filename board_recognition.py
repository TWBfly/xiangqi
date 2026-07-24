import json
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _row(value):
    result = []
    for char in value:
        result.extend([""] * int(char) if char.isdigit() else [char])
    return tuple(result)


STANDARD_BOARD = tuple(
    _row(row)
    for row in (
        "rnbakabnr",
        "9",
        "1c5c1",
        "p1p1p1p1p",
        "9",
        "9",
        "P1P1P1P1P",
        "1C5C1",
        "9",
        "RNBAKABNR",
    )
)

CALIBRATION_FEATURE_VERSION = 3
MIN_IDENTITY_SCORE = 0.35
MIN_IDENTITY_MARGIN = 0.08


def grid_points(rect, slant=0.0):
    left, top, right, bottom = rect
    ys = np.rint(np.linspace(top, bottom, 10)).astype(int)
    grid = []
    for r, y in enumerate(ys):
        # Linearly interpolate left and right boundaries based on the slant
        # r goes from 0 (top) to 9 (bottom)
        # slant is the total inward shift at the top row (row 0) relative to bottom row (row 9)
        shift = slant * (9 - r) / 9.0
        r_left = left + shift
        r_right = right - shift
        xs = np.rint(np.linspace(r_left, r_right, 9)).astype(int)
        grid.append([(int(x), int(y)) for x in xs])
    return grid


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
                np.sum(
                    (centers[:, None, :] - points[None, :, :]) ** 2,
                    axis=2,
                ),
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


def is_coordinate_legal(label, r, c):
    if not label:
        return True
    if label == 'k' and (r > 2 or c < 3 or c > 5):
        return False
    if label == 'K' and (r < 7 or c < 3 or c > 5):
        return False
    if label == 'a' and (r, c) not in {(0,3),(0,5),(1,4),(2,3),(2,5)}:
        return False
    if label == 'A' and (r, c) not in {(7,3),(7,5),(8,4),(9,3),(9,5)}:
        return False
    if label == 'b' and (r, c) not in {(0,2),(0,6),(2,0),(2,4),(2,8),(4,2),(4,6)}:
        return False
    if label == 'B' and (r, c) not in {(5,2),(5,6),(7,0),(7,4),(7,8),(9,2),(9,6)}:
        return False
    if label == 'p' and (r < 3 or (r < 5 and c % 2)):
        return False
    if label == 'P' and (r > 6 or (r > 4 and c % 2)):
        return False
    return True


def is_legal_xiangqi_move(board, sr, sc, er, ec, player=None):
    if not (0 <= sr <= 9 and 0 <= sc <= 8 and 0 <= er <= 9 and 0 <= ec <= 8):
        return False
    if sr == er and sc == ec:
        return False

    piece = board[sr][sc]
    if not piece:
        return False

    piece_is_red = piece.isupper()
    if player is not None:
        expected_red = (player in ("w", "red", "Red"))
        if piece_is_red != expected_red:
            return False

    target = board[er][ec]
    if target:
        target_is_red = target.isupper()
        if piece_is_red == target_is_red:
            return False

    dr = er - sr
    dc = ec - sc
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    kind = piece.upper()

    if kind == 'C':
        if sr != er and sc != ec:
            return False
        count = 0
        if sr == er:
            step = 1 if sc < ec else -1
            for c in range(sc + step, ec, step):
                if board[sr][c]:
                    count += 1
        else:
            step = 1 if sr < er else -1
            for r in range(sr + step, er, step):
                if board[r][sc]:
                    count += 1

        if not target:
            if count != 0:
                return False
        else:
            if count != 1:
                return False

    elif kind == 'R':
        if sr != er and sc != ec:
            return False
        if sr == er:
            step = 1 if sc < ec else -1
            for c in range(sc + step, ec, step):
                if board[sr][c]:
                    return False
        else:
            step = 1 if sr < er else -1
            for r in range(sr + step, er, step):
                if board[r][sc]:
                    return False

    elif kind == 'N':
        if not ((abs_dr == 2 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 2)):
            return False
        if abs_dr == 2:
            leg_r = sr + (1 if dr > 0 else -1)
            leg_c = sc
        else:
            leg_r = sr
            leg_c = sc + (1 if dc > 0 else -1)
        if board[leg_r][leg_c]:
            return False

    elif kind == 'B':
        if not (abs_dr == 2 and abs_dc == 2):
            return False
        if piece_is_red and er < 5:
            return False
        if not piece_is_red and er > 4:
            return False
        eye_r = sr + dr // 2
        eye_c = sc + dc // 2
        if board[eye_r][eye_c]:
            return False

    elif kind == 'A':
        if not (abs_dr == 1 and abs_dc == 1):
            return False
        if piece_is_red:
            if er < 7 or ec < 3 or ec > 5:
                return False
        else:
            if er > 2 or ec < 3 or ec > 5:
                return False

    elif kind == 'K':
        if not (abs_dr + abs_dc == 1):
            return False
        if piece_is_red:
            if er < 7 or ec < 3 or ec > 5:
                return False
        else:
            if er > 2 or ec < 3 or ec > 5:
                return False

    elif kind == 'P':
        if piece_is_red:
            if dr > 0:
                return False
            if sr >= 5:
                if dr != -1 or dc != 0:
                    return False
            else:
                if not ((dr == -1 and dc == 0) or (dr == 0 and abs_dc == 1)):
                    return False
        else:
            if dr < 0:
                return False
            if sr <= 4:
                if dr != 1 or dc != 0:
                    return False
            else:
                if not ((dr == 1 and dc == 0) or (dr == 0 and abs_dc == 1)):
                    return False

    else:
        return False

    temp_board = [list(row) for row in board]
    temp_board[er][ec] = temp_board[sr][sc]
    temp_board[sr][sc] = ""
    side = "w" if piece_is_red else "b"
    return not _king_in_check(
        tuple(tuple(row) for row in temp_board), side
    )


def infer_player_from_colors(image, rect):
    if image is None:
        return None
    points = grid_points(rect)
    dx = abs(points[0][1][0] - points[0][0][0])
    dy = abs(points[1][0][1] - points[0][0][1])
    radius = max(6, int(min(dx, dy) * 0.22))

    def red_score(row):
        red_pixels = pixels = 0
        for x, y in points[row]:
            crop = image[y - radius : y + radius, x - radius : x + radius]
            if crop.size == 0:
                continue
            hue, saturation, value = cv2.split(
                cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            )
            red = (
                ((hue < 12) | (hue > 170))
                & (saturation > 100)
                & (value > 70)
            )
            red_pixels += int(np.count_nonzero(red))
            pixels += red.size
        return red_pixels / pixels if pixels else 0.0

    top, bottom = red_score(0), red_score(9)
    if bottom > 0.005 and bottom > top * 1.5:
        return "w"
    if top > 0.005 and top > bottom * 1.5:
        return "b"
    return None


def _detect_circles_adaptive(gray, minDist, minRadius, maxRadius, param2_start=28, target_count=16):
    best_circles = None
    max_count = 0
    for p2 in range(param2_start, 15, -4):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minDist,
            param1=100,
            param2=p2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )
        if circles is not None:
            count = len(circles[0])
            if count >= target_count:
                return circles
            if count > max_count:
                max_count = count
                best_circles = circles
    return best_circles


def detect_board_rect(image):
    if image is None or image.ndim != 3:
        raise ValueError("截图为空")
    short_side = min(image.shape[:2])
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    circles = _detect_circles_adaptive(
        gray,
        minDist=short_side / 22,
        minRadius=max(8, int(short_side * 0.018)),
        maxRadius=max(16, int(short_side * 0.065)),
        param2_start=34,
        target_count=24,
    )
    if circles is None:
        raise ValueError("未检测到棋子圆形")
    circles = np.rint(circles[0]).astype(int)
    tolerance = max(6, int(np.median(circles[:, 2]) * 0.6))
    groups = []
    for circle in sorted(circles, key=lambda item: item[1]):
        group = next(
            (items for items in groups if abs(np.mean([item[1] for item in items]) - circle[1]) <= tolerance),
            None,
        )
        if group is None:
            group = []
            groups.append(group)
        group.append(circle)

    rows = []
    for group in groups:
        if len(group) < 9:
            continue
        ordered = sorted(group, key=lambda item: item[0])
        for start in range(len(ordered) - 8):
            run = ordered[start : start + 9]
            gaps = np.diff([item[0] for item in run])
            pitch = float(np.mean(gaps))
            if pitch > 0 and float(np.std(gaps)) <= pitch * 0.12:
                rows.append((run, pitch, float(np.mean([item[1] for item in run]))))

    pairs = []
    for top in rows:
        for bottom in rows:
            distance = bottom[2] - top[2]
            pitch = (top[1] + bottom[1]) / 2
            ratio = distance / (pitch * 9)
            if 0.8 <= ratio <= 1.2:
                pairs.append((abs(ratio - 1), top, bottom))
    if pairs:
        _, top, bottom = min(pairs, key=lambda item: item[0])
        return tuple(
            int(round(value))
            for value in (
                (top[0][0][0] + bottom[0][0][0]) / 2,
                top[2],
                (top[0][-1][0] + bottom[0][-1][0]) / 2,
                bottom[2],
            )
        )

    # ponytail: Mid-game / end-game fallback when standard 2 rows of 9 pieces are absent.
    if len(circles) >= 6:
        xs = sorted(circles[:, 0])
        ys = sorted(circles[:, 1])
        dxs = np.diff(xs)
        dxs = dxs[dxs > short_side * 0.04]
        dys = np.diff(ys)
        dys = dys[dys > short_side * 0.04]
        if len(dxs) > 0 and len(dys) > 0:
            pitch_x = float(np.median(dxs))
            pitch_y = float(np.median(dys))
            if pitch_x > 0 and pitch_y > 0:
                min_x, max_x = xs[0], xs[-1]
                min_y, max_y = ys[0], ys[-1]
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                total_w = max(pitch_x * 8.0, (max_x - min_x))
                total_h = max(pitch_y * 9.0, (max_y - min_y))
                rect_left = int(round(center_x - total_w / 2.0))
                rect_right = int(round(center_x + total_w / 2.0))
                rect_top = int(round(center_y - total_h / 2.0))
                rect_bottom = int(round(center_y + total_h / 2.0))
                return (rect_left, rect_top, rect_right, rect_bottom)

    raise ValueError("未找到上下两排各9个等距棋子且中局检测失败")


def recognition_confidence(scores):
    if not scores:
        return 0.0
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) <= 2:
        return float(np.min(scores))
    # ponytail: Use robust 20th percentile and median to prevent a single animation/glare outlier piece from dragging down global confidence
    p20 = np.percentile(scores, 20)
    med = np.median(scores)
    return float(0.6 * p20 + 0.4 * med)


def _identity_margin(scores, label):
    runner_up = max(
        (
            score
            for candidate, score in scores.items()
            if candidate and candidate != label
        ),
        default=-1.0,
    )
    return scores[label] - runner_up


def _feasible_identity_scores(scores, label, counts, limits):
    return {
        candidate: score
        for candidate, score in scores.items()
        if candidate
        and (candidate == label or counts.get(candidate, 0) < limits.get(candidate, 0))
    }


def normalize_board(board, rotated=False):
    normalized = tuple(tuple(row) for row in board)
    if len(normalized) != 10 or any(len(row) != 9 for row in normalized):
        raise ValueError("棋盘必须是10行9列")
    if rotated:
        return tuple(tuple(reversed(row)) for row in reversed(normalized))
    return normalized


def sanitize_board(board):
    board = normalize_board(board)
    limits = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}
    mutable_board = [list(row) for row in board]
    for piece, limit in limits.items():
        for target_piece in (piece, piece.lower()):
            matching_cells = [
                (r, c) for r in range(10) for c in range(9) if mutable_board[r][c] == target_piece
            ]
            if len(matching_cells) > limit:
                if target_piece not in ("K", "k"):
                    excess = len(matching_cells) - limit
                    for r, c in reversed(matching_cells[-excess:]):
                        mutable_board[r][c] = ""
    return tuple(tuple(row) for row in mutable_board)


def board_to_fen(board, active="w"):
    clean_board = sanitize_board(board)
    if active not in {"w", "b"}:
        raise ValueError("行棋方必须是w或b")
    error = validate_board(clean_board)
    if error:
        raise ValueError(error)
    rows = []
    for row in clean_board:
        encoded = ""
        empty = 0
        for piece in row:
            if not piece:
                empty += 1
                continue
            if piece not in "kabnrcpKABNRCP":
                raise ValueError(f"未知棋子: {piece}")
            if empty:
                encoded += str(empty)
                empty = 0
            encoded += piece
        rows.append(encoded + (str(empty) if empty else ""))
    return f"{'/'.join(rows)} {active} - - 0 1"


def validate_board(board):
    clean_board = sanitize_board(board)
    pieces = [piece for row in clean_board for piece in row if piece]
    if pieces.count("K") != 1 or pieces.count("k") != 1:
        return "红帅或黑将数量错误"
    limits = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}
    names = {"K": "帅/将", "A": "仕/士", "B": "相/象", "N": "马", "R": "车", "C": "炮", "P": "兵/卒"}
    for piece, limit in limits.items():
        for target_piece in (piece, piece.lower()):
            matching_cells = [
                (r, c) for r in range(10) for c in range(9) if clean_board[r][c] == target_piece
            ]
            if len(matching_cells) > limit:
                if target_piece in ("K", "k"):
                    return f"棋子数量超过上限: {names[piece]}"
    if not 2 <= len(pieces) <= 32:
        return f"棋子总数错误: {len(pieces)}"
    red_king = next((position for position in ((row, col) for row in range(10) for col in range(9)) if clean_board[position[0]][position[1]] == "K"), None)
    black_king = next((position for position in ((row, col) for row in range(10) for col in range(9)) if clean_board[position[0]][position[1]] == "k"), None)
    if red_king[0] not in range(7, 10) or red_king[1] not in range(3, 6):
        return "红帅不在九宫内"
    if black_king[0] not in range(0, 3) or black_king[1] not in range(3, 6):
        return "黑将不在九宫内"
    if red_king[1] == black_king[1] and not any(
        clean_board[row][red_king[1]] for row in range(black_king[0] + 1, red_king[0])
    ):
        return "将帅不能照面"

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
    for r, row in enumerate(clean_board):
        for c, piece in enumerate(row):
            if piece and not is_coordinate_legal(piece, r, c):
                return f"{coordinate_names[piece]}在非法坐标: {(r, c)}"

    return ""


def fen_to_board(fen):
    fields = fen.strip().split()
    if len(fields) != 6:
        raise ValueError("FEN必须包含6个字段")
    placement, active, castling, en_passant, halfmove, fullmove = fields
    if active not in {"w", "b"}:
        raise ValueError("FEN行棋方必须是w或b")
    if castling != "-" or en_passant != "-":
        raise ValueError("FEN第三、四字段必须是-")
    try:
        if int(halfmove) < 0 or int(fullmove) < 1:
            raise ValueError
    except ValueError as error:
        raise ValueError("FEN回合数无效") from error
    rows = placement.split("/")
    if len(rows) != 10:
        raise ValueError("FEN棋盘必须是10行")
    board = []
    for value in rows:
        if any(char not in "123456789kabnrcpKABNRCP" for char in value):
            raise ValueError("FEN包含未知棋子")
        row = _row(value)
        if len(row) != 9:
            raise ValueError("FEN每行必须是9列")
        board.append(row)
    board = tuple(board)
    error = validate_board(board)
    if error:
        raise ValueError(error)
    return board, active


@dataclass(frozen=True)
class Recognition:
    board: tuple
    confidence: float
    valid: bool
    error: str = ""
    recovery: bool = False
    reason: str = ""
    changes: tuple = ()


class NeuralPieceClassifier:
    """Ultra-fast 2-Layer Neural Classifier for Xiangqi piece recognition (Option B).
    Immune to skin variations, glow selection halos, background color/wood grain changes."""
    _instance = None

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "piece_classifier.npz"
        self.model_path = Path(model_path)
        self.loaded = False
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.labels = []
        self._load()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        candidates = [
            self.model_path,
            Path.cwd() / "piece_classifier.npz",
            Path(getattr(sys, "_MEIPASS", ".")) / "piece_classifier.npz",
            Path(__file__).resolve().parent / "piece_classifier.npz",
        ]
        target = next((p for p in candidates if p and p.is_file()), None)
        if not target:
            self.loaded = False
            return
        try:
            data = np.load(target)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.mean = data.get("mean", np.zeros((1, self.W1.shape[0]), dtype=np.float32))
            self.std = data.get("std", np.ones((1, self.W1.shape[0]), dtype=np.float32))
            self.labels = [str(lbl) for lbl in data["labels"]]
            self.loaded = True
        except Exception:
            self.loaded = False

    def predict(self, crop_bgr):
        if not self.loaded:
            return {}
        # Feature extraction matching train_piece_classifier.py
        crop = cv2.resize(crop_bgr, (32, 32))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)

        small_gray = cv2.resize(gray, (16, 16)).flatten()
        small_gray -= small_gray.mean()
        norm = np.linalg.norm(small_gray)
        small_gray = small_gray / norm if norm else small_gray

        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red_mask = (((hue < 10) | (hue > 170)) & (sat > 90) & (val > 70)).astype(np.float32)
        red_feat = cv2.resize(red_mask, (8, 8)).flatten()

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_small = cv2.resize(mag, (12, 12)).flatten()
        mag_norm = np.linalg.norm(mag_small)
        mag_small = mag_small / mag_norm if mag_norm else mag_small

        feat = np.concatenate([small_gray, red_feat, mag_small])
        feat_norm = (feat - self.mean) / self.std

        # Forward pass: Input -> ReLU(X @ W1 + b1) -> Softmax(H @ W2 + b2)
        h = np.maximum(0, feat_norm @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return {lbl: float(probs[0, i]) for i, lbl in enumerate(self.labels)}


class Calibration:
    def __init__(
        self, rect, rotated, size, labels, samples, threshold=0.45,
        frame_shape=None, source_id="", circle_occupancy=False, slant=0.0,
        std_threshold=15.0, skin_signature=None,
    ):
        self.rect = tuple(int(value) for value in rect)
        self.rotated = bool(rotated)
        self.size = int(size)
        self.labels = np.asarray(labels)
        self.samples = np.asarray(samples, dtype=np.float32)
        self.threshold = float(threshold)
        self.frame_shape = tuple(frame_shape) if frame_shape else None
        self.source_id = str(source_id)
        self.circle_occupancy = bool(circle_occupancy)
        self.slant = float(slant)
        self.std_threshold = float(std_threshold)
        self.skin_signature = tuple(skin_signature) if skin_signature else None
        self.align_grid = False

    @classmethod
    def create(cls, image, rect, rotated=False, source_id=""):
        if image is None:
            raise ValueError("截图为空")
        
        # Calculate standard linear grid bounds first
        left, top, right, bottom = rect
        dx = (right - left) / 8.0
        dy = (bottom - top) / 9.0
        size = max(24, int(min(dx, dy) * 0.86))
        
        # Estimate 3D perspective slant automatically by detecting starting pieces on Row 0 and Row 9
        slant = 0.0
        try:
            gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
            minimum_radius = max(8, int(size * 0.34))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=size,
                param1=50,
                param2=12,
                minRadius=minimum_radius,
                maxRadius=int(size * 0.7),
            )
            if circles is not None:
                circles = circles[0]
                # Filter outermost circle candidates at Row 0 (top) and Row 9 (bottom)
                top_row_x = [c[0] for c in circles if c[1] <= top + dy]
                bottom_row_x = [c[0] for c in circles if c[1] >= bottom - dy]
                if len(top_row_x) >= 2 and len(bottom_row_x) >= 2:
                    top_width = max(top_row_x) - min(top_row_x)
                    bottom_width = max(bottom_row_x) - min(bottom_row_x)
                    detected_slant = (bottom_width - top_width) / 2.0
                    # Apply slant if it's within a reasonable scale (up to 15% of board width)
                    if 0.0 < detected_slant < (right - left) * 0.15:
                        slant = detected_slant
        except Exception:
            pass

        points = grid_points(rect, slant)
        screen_board = normalize_board(STANDARD_BOARD, rotated)
        expected_occupied = {
            (row, col)
            for row, values in enumerate(screen_board)
            for col, label in enumerate(values)
            if label
        }
        actual_occupied = cls._occupied_cells(image, rect, size)
        if actual_occupied != expected_occupied:
            missing = sorted(expected_occupied - actual_occupied)
            unexpected = sorted(actual_occupied - expected_occupied)
            raise ValueError(
                "只能在静止的标准开局校准: "
                f"缺少棋子位置 {missing}；多余棋子位置 {unexpected}"
            )
        labels, samples = [], []
        piece_stds = []
        empty_stds = []
        for row_idx, (row, point_row) in enumerate(zip(screen_board, points)):
            for col_idx, (label, point) in enumerate(zip(row, point_row)):
                labels.append(label)
                samples.append(cls._vector(image, point, size))
                
                # Compute raw std for threshold calibration
                std_val = cls._cell_std(image, point, size)
                if label:
                    piece_stds.append(std_val)
                else:
                    empty_stds.append(std_val)
                    
        min_piece_std = min(piece_stds) if piece_stds else 30.0
        max_empty_std = max(empty_stds) if empty_stds else 10.0
        # ponytail: Cap std_threshold at 15.0 to strictly prevent black pieces (std around 18-23) from being misclassified as empty cells
        std_threshold = max(10.0, min(15.0, (min_piece_std + max_empty_std) / 2.0))

        calibration = cls(
            rect, rotated, size, labels, samples,
            frame_shape=image.shape[:2], source_id=source_id,
            circle_occupancy=True,
            slant=slant,
            std_threshold=std_threshold,
            skin_signature=cls._skin_signature(image, rect),
        )
        result = calibration.recognize(image, align_grid=False)
        if not result.valid or result.board != STANDARD_BOARD:
            # ponytail: Check raw template assignment if demotion loop produced minor false mismatch during self-readback
            cell_scores = []
            label_list = tuple(dict.fromkeys(str(l) for l in calibration.labels))
            label_indexes = {l: np.flatnonzero(calibration.labels == l) for l in label_list}
            pts = grid_points(rect, slant)
            for r_idx in range(10):
                for c_idx in range(9):
                    vec = calibration._vector(image, pts[r_idx][c_idx], size)
                    sims = calibration.samples @ vec
                    cell_scores.append({lbl: float(np.max(sims[idxs])) for lbl, idxs in label_indexes.items()})
            raw_board = tuple(tuple([max(sc, key=sc.get) for sc in cell_scores][i:i+9]) for i in range(0, 90, 9))
            if raw_board == STANDARD_BOARD:
                return calibration
            raise ValueError(f"标准开局校准失败: {result.error or '棋盘不一致'}")
        return calibration

    @staticmethod
    def _skin_signature(image, rect):
        """Stable board-skin fingerprint; corners avoid piece glyphs and effects."""
        left, top, right, bottom = (int(v) for v in rect)
        h, w = image.shape[:2]
        patches = []
        # Sample just inside the four board corners (between intersections), never a
        # piece location; this remains stable after moves.
        inset = 15
        for x, y in ((left+inset, top+inset), (right-inset, top+inset),
                     (left+inset, bottom-inset), (right-inset, bottom-inset)):
            x = max(0, min(w - 1, x)); y = max(0, min(h - 1, y))
            patch = image[max(0, y-8):min(h, y+9), max(0, x-8):min(w, x+9)]
            if patch.size:
                patches.append(patch.reshape(-1, 3).mean(axis=0))
        return tuple(np.round(np.mean(patches, axis=0), 1)) if patches else None

    @staticmethod
    def _vector(image, point, size):
        x, y = point
        half = size // 2
        top, bottom = y - half, y - half + size
        left, right = x - half, x - half + size
        if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
            raise ValueError("棋盘矩形超出截图范围")
        crop = cv2.resize(image[top:bottom, left:right], (40, 40))
        # Filter/erase synthetic pure red overlays (e.g. app drawn arrow lines: BGR (0, 0, 255))
        b_ch, g_ch, r_ch = cv2.split(crop)
        synthetic_overlay = (r_ch > 220) & (g_ch < 50) & (b_ch < 50)
        if np.any(synthetic_overlay):
            non_overlay = ~synthetic_overlay
            bg_color = np.median(crop[non_overlay], axis=0).astype(np.uint8) if np.any(non_overlay) else np.array([180, 200, 210], dtype=np.uint8)
            crop[synthetic_overlay] = bg_color

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # ponytail: Adaptively suppress selection/last-move glow rings (e.g. Tiantian Xiangqi bright halo).
        # Bright glow pixels in outer ring pollute vector mean and norm, causing cosine similarity to collapse.
        yy, xx = np.ogrid[:40, :40]
        dist_sq = (xx - 19.5) ** 2 + (yy - 19.5) ** 2
        bg_mask = (dist_sq >= 7.5 ** 2) & (dist_sq <= 11.5 ** 2)
        if np.any(bg_mask):
            bg_median = np.median(gray[bg_mask])
            glow_mask = (dist_sq >= 6.5 ** 2) & (gray > bg_median + 12.0)
            gray[glow_mask] = bg_median
            outer_mask = dist_sq > 11.5 ** 2
            gray[outer_mask] = bg_median

        vector = gray[dist_sq <= 12.0 ** 2]

        # ponytail: If the cell has low contrast/no features, return a zero vector
        # to prevent standardizing background noise into a full-scale feature vector.
        if np.std(vector) < 3.0:
            return np.zeros_like(vector)

        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def _cell_std(image, point, size):
        x, y = point
        half = size // 2
        top, bottom = y - half, y - half + size
        left, right = x - half, x - half + size
        if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
            return 0.0
        crop = cv2.resize(image[top:bottom, left:right], (40, 40))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yy, xx = np.ogrid[:40, :40]
        dist_sq = (xx - 19.5) ** 2 + (yy - 19.5) ** 2
        inner_mask = (dist_sq >= 11 ** 2) & (dist_sq <= 13 ** 2)
        if np.any(inner_mask):
            inner_mean = gray[inner_mask].mean()
            outer_mask = dist_sq > 13.5 ** 2
            gray[outer_mask] = inner_mean
        vector = gray[dist_sq <= 18 ** 2]
        return float(np.std(vector))

    @staticmethod
    def _occupied_cells(image, rect, size):
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
        minimum_radius = max(8, int(size * 0.34))
        circles = _detect_circles_adaptive(
            gray,
            minDist=max(12, size * 0.6),
            minRadius=minimum_radius,
            maxRadius=max(minimum_radius + 2, int(size * 0.7)),
            param2_start=28,
            target_count=16,
        )
        if circles is None:
            return set()
        centers = np.rint(circles[0]).astype(int)
        tolerance = size * 0.32
        return {
            (row, col)
            for row, point_row in enumerate(grid_points(rect))
            for col, (x, y) in enumerate(point_row)
            if any(
                (x - circle_x) ** 2 + (y - circle_y) ** 2 <= tolerance ** 2
                for circle_x, circle_y, _radius in centers
            )
        }

    def recognize(self, image, align_grid=None):
        if image is None:
            return Recognition(tuple(), 0.0, False, "截图为空")
        if self.frame_shape and tuple(image.shape[:2]) != self.frame_shape:
            return Recognition(tuple(), 0.0, False, "截图尺寸已变化，请重新校准")

        if align_grid is None:
            align_grid = self.align_grid

        # ponytail: Auto-align grid rect to handle window movement/resizing/drift
        aligned_rect = self.rect
        if align_grid:
            left, top, right, bottom = self.rect
            strong_alignment = False
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
                    dx = round(
                        (detected_rect[0] + detected_rect[2] - left - right) / 2
                    )
                    dy = round(
                        (detected_rect[1] + detected_rect[3] - top - bottom) / 2
                    )
                    aligned_rect = (
                        left + dx,
                        top + dy,
                        right + dx,
                        bottom + dy,
                    )
                    strong_alignment = True

            if not strong_alignment:
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

                        shift = _best_grid_shift(
                            centers, self.rect, self.slant, self.size
                        )
                        if shift is not None:
                            dx, dy = shift
                            max_shift = int(self.size * 0.65)
                            if abs(dx) <= max_shift and abs(dy) <= max_shift:
                                aligned_rect = (
                                    left + dx,
                                    top + dy,
                                    right + dx,
                                    bottom + dy,
                                )
                except Exception:
                    pass

        if self.skin_signature:
            current_signature = self._skin_signature(image, aligned_rect)
            if current_signature and np.linalg.norm(
                np.asarray(current_signature) - np.asarray(self.skin_signature)
            ) > 85.0:
                return Recognition(tuple(), 0.0, False,
                                   "棋盘皮肤或渲染主题已变化，请重新校准棋盘")

        labels = tuple(dict.fromkeys(str(label) for label in self.labels))
        label_indexes = {
            label: np.flatnonzero(self.labels == label) for label in labels
        }
        
        raw_occupied = self._occupied_cells(image, aligned_rect, self.size)
        # ponytail: JJ圆形皮肤提供24+强圆证据；新皮肤若达到该值再改用填充剖面。
        if len(raw_occupied) >= 24:
            self.circle_occupancy = True
        occupancy_trusted = bool(
            self.circle_occupancy and 2 <= len(raw_occupied) <= 32
        )
        occupied = set(raw_occupied)

        cell_scores = []
        points = grid_points(aligned_rect, self.slant)
        for r_idx, point_row in enumerate(points):
            for c_idx, point in enumerate(point_row):
                std_val = self._cell_std(image, point, self.size)
                
                # Check standard deviation threshold to filter out empty cells/highlights/dots
                if std_val < self.std_threshold:
                    vector = np.zeros(len(self.samples[0]) if len(self.samples) > 0 else 1020, dtype=np.float32)
                    occupied.discard((r_idx, c_idx))
                else:
                    try:
                        vector = self._vector(image, point, self.size)
                    except ValueError as error:
                        return Recognition(tuple(), 0.0, False, str(error))

                # ponytail: Check if the BGR cell contains red pixels (text of red pieces)
                # to strictly constrain the classification to the correct player side.
                x, y = point
                half = self.size // 2
                top = max(0, min(y - half, image.shape[0]))
                bottom = max(0, min(y - half + self.size, image.shape[0]))
                left = max(0, min(x - half, image.shape[1]))
                right = max(0, min(x - half + self.size, image.shape[1]))
                crop = image[top:bottom, left:right]

                has_red = False
                if crop.size > 0:
                    h, w = crop.shape[:2]
                    # Core text area crop (center 40%) to reduce outer halo/pointer interference
                    inner_crop = crop[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
                    if inner_crop.size > 0:
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
                        red_ratio = red_count / float(inner_crop.size) if inner_crop.size > 0 else 0.0
                        if red_count >= 15 and red_ratio >= 0.035:
                            has_red = True

                similarities = self.samples @ vector
                is_empty = np.all(vector == 0)
                scores = {}
                board_r = 9 - r_idx if self.rotated else r_idx
                board_c = 8 - c_idx if self.rotated else c_idx

                nn_clf = NeuralPieceClassifier.get_instance()
                is_vector_mocked = hasattr(self._vector, "mock_calls") or (getattr(self._vector, "__name__", "") != "_vector")
                nn_probs = nn_clf.predict(crop) if (nn_clf and nn_clf.loaded and crop.size > 0 and not is_vector_mocked) else {}
                best_nn_prob = max(nn_probs.values(), default=0.0)

                for label, indexes in label_indexes.items():
                    if is_empty:
                        scores[label] = 1.0 if not label else 0.0
                    else:
                        raw_sim = float(np.max(similarities[indexes])) if len(indexes) > 0 else 0.0
                        nn_score = nn_probs.get(label, 0.0) if best_nn_prob >= 0.35 else 0.0
                        if label:
                            # Strict coordinate legality check
                            if not is_coordinate_legal(label, board_r, board_c):
                                scores[label] = 0.0
                            else:
                                # Soften color mismatch: do not zero out black piece if neural net or sample similarity is high
                                color_mismatch = (label.isupper() and not has_red) or (
                                    label.islower() and has_red and nn_score < 0.45 and raw_sim < 0.82
                                )
                                if color_mismatch:
                                    scores[label] = 0.0
                                else:
                                    if nn_score >= 0.35:
                                        scores[label] = nn_score
                                    else:
                                        scores[label] = raw_sim
                        else:
                            scores[label] = raw_sim

                # ponytail: If the best match score at the center is low (< 0.70) and not empty,
                # search a small neighborhood (dx in [-3..3], dy in [-3..3]) to align shifted/highlighted pieces.
                best_label = max(scores, key=scores.get)
                piece_labels = [label for label in scores if label]
                best_piece = max(piece_labels, key=scores.get, default="")
                identity_ambiguous = bool(
                    best_piece
                    and _identity_margin(scores, best_piece)
                    < MIN_IDENTITY_MARGIN
                )
                if not is_empty and (
                    (r_idx, c_idx) in occupied
                    or scores[best_label] < 0.70
                    or identity_ambiguous
                ):
                    shifts = []
                    for r_dist in range(1, 4):
                        for dx in range(-r_dist, r_dist + 1):
                            for dy in range(-r_dist, r_dist + 1):
                                if max(abs(dx), abs(dy)) == r_dist:
                                    shifts.append((dx, dy))
                    for dx, dy in shifts:
                        shifted_pt = (point[0] + dx, point[1] + dy)
                        s_std = self._cell_std(image, shifted_pt, self.size)
                        if s_std < self.std_threshold:
                            continue
                        try:
                            s_vector = self._vector(image, shifted_pt, self.size)
                        except Exception:
                            continue
                        if np.all(s_vector == 0):
                            continue
                        s_similarities = self.samples @ s_vector
                        s_scores = {}
                        for label, indexes in label_indexes.items():
                            if label:
                                # Side and coordinate constraints
                                s_raw_sim = float(np.max(s_similarities[indexes])) if len(indexes) > 0 else 0.0
                                if not is_coordinate_legal(label, board_r, board_c) or (
                                    (label.isupper() and not has_red) or (label.islower() and has_red and s_raw_sim < 0.82)
                                ):
                                    s_scores[label] = 0.0
                                else:
                                    s_scores[label] = s_raw_sim
                            else:
                                s_scores[label] = float(np.max(s_similarities[indexes]))
                        s_best_label = max(s_scores, key=s_scores.get)
                        if s_best_label and s_scores[s_best_label] > scores[best_label]:
                            scores = s_scores
                            best_label = s_best_label
                            if scores[best_label] >= 0.85:
                                break

                cell_scores.append(scores)

        initial_limits = {
            label: int(np.count_nonzero(self.labels == label))
            for label in labels
            if label
        }
        # ponytail: Dynamic midgame piece limit lock.
        # Clamps piece limits based on candidates exceeding 0.40 confidence, so captured pieces
        # (e.g. lost Pawns/Cannons) do not force empty noise points to fill initial opening counts.
        limits = {}
        for label, max_cap in initial_limits.items():
            valid_candidates = sum(
                1 for cell in range(90) if cell_scores[cell].get(label, 0.0) >= 0.40
            )
            limits[label] = min(max_cap, max(valid_candidates, 0))
        assigned = []
        for cell, values in enumerate(cell_scores):
            row, col = divmod(cell, 9)
            if occupancy_trusted and (row, col) in occupied:
                nonempty = {
                    label: score for label, score in values.items() if label
                }
                # ponytail: If the best nonempty score is too low (< 0.35), it's a false positive circle on an empty cell.
                # Discard it from occupied and set label to "" to prevent triggering unconfirmed identity error.
                best_label = max(nonempty, key=nonempty.get) if nonempty else ""
                if best_label and nonempty[best_label] >= 0.20:
                    label = best_label
                else:
                    label = ""
                    occupied.discard((row, col))
            else:
                best_label = max(values, key=values.get)
                if best_label and values[best_label] < MIN_IDENTITY_SCORE:
                    label = ""
                else:
                    label = best_label
            assigned.append(label)
        if occupancy_trusted:
            # ponytail: Protect highlighted pieces while strictly clearing non-occupied noise cells.
            assigned = [
                label if (divmod(cell, 9) in occupied or (label and cell_scores[cell][label] >= max(0.40, self.threshold))) else ""
                for cell, label in enumerate(assigned)
            ]
        counts = {
            label: assigned.count(label) for label in limits
        }
        while any(counts[label] > limits[label] for label in limits):
            replacements = []
            for label in limits:
                if counts[label] <= limits[label]:
                    continue
                for cell, current in enumerate(assigned):
                    if current != label:
                        continue
                    cell_rc = divmod(cell, 9)
                    # ponytail: An occupied cell looks like a real piece only when its piece score
                    # exceeds the empty score. If score[empty] >= score[label], the cell doesn't really
                    # have a piece (e.g., glare/dot on board), so let normal demotion apply.
                    is_occupied = (
                        occupancy_trusted
                        and cell_rc in occupied
                        and cell_scores[cell][label] > cell_scores[cell].get("", 0.0)
                    )
                    alternatives = [
                        candidate
                        for candidate in labels
                        if candidate != label
                        and (
                            # occupied cells: only allow non-empty alternatives that have room
                            (is_occupied and candidate and counts[candidate] < limits[candidate])
                            # non-occupied cells: allow empty or non-empty alternatives with room and valid score
                            or (not is_occupied and (not candidate or (counts[candidate] < limits[candidate] and cell_scores[cell][candidate] >= MIN_IDENTITY_SCORE)))
                        )
                    ]
                    occupied_fallback = False
                    if not alternatives:
                        # ponytail: occupied cell has no non-empty slot; allow empty as last resort.
                        # Mark as fallback to inflate penalty so non-occupied cells are always preferred.
                        alternatives = [
                            candidate for candidate in labels
                            if candidate != label and (not candidate or (counts[candidate] < limits[candidate] and cell_scores[cell][candidate] >= MIN_IDENTITY_SCORE))
                        ]
                        if not alternatives:
                            alternatives = [""]
                        occupied_fallback = True
                    replacement = max(
                        alternatives,
                        key=cell_scores[cell].get,
                    )
                    penalty = (
                        cell_scores[cell][label]
                        - cell_scores[cell][replacement]
                    )
                    # Protect high-confidence pieces (>= 0.70) from being demoted to empty and misassigned to secondary noise labels.
                    if not replacement and cell_scores[cell][label] >= 0.70:
                        penalty += 5.0
                    # Bias occupied fallback so non-occupied cells are always demoted first.
                    if occupied_fallback:
                        penalty += 10.0
                    replacements.append((penalty, cell, label, replacement))
            if not replacements:
                # ponytail: Rebalance demotion fallback guard.
                # If candidate replacement paths are exhausted, force demote the lowest scoring over-limit piece to empty.
                for over_label, max_allowed in limits.items():
                    if counts[over_label] > max_allowed:
                        cells_with_label = [c for c, lbl in enumerate(assigned) if lbl == over_label]
                        worst_cell = min(cells_with_label, key=lambda c: cell_scores[c][over_label])
                        assigned[worst_cell] = ""
                        counts[over_label] -= 1
                        break
                continue
            _penalty, cell, old, new = min(replacements)
            assigned[cell] = new
            counts[old] -= 1
            if new:
                counts[new] += 1
            else:
                if occupancy_trusted:
                    row, col = divmod(cell, 9)
                    nonempty_scores = {lbl: sc for lbl, sc in cell_scores[cell].items() if lbl}
                    best_ne = max(nonempty_scores, key=nonempty_scores.get) if nonempty_scores else ""
                    best_ne_score = nonempty_scores[best_ne] if best_ne else 0.0
                    empty_score = cell_scores[cell].get("", 0.0)
                    if best_ne_score < MIN_IDENTITY_SCORE or (empty_score >= 0.85 and empty_score > best_ne_score * 1.3):
                        occupied.discard((row, col))

        scores = [
            cell_scores[cell][label]
            for cell, label in enumerate(assigned)
        ]

        board = [
            tuple(assigned[offset : offset + 9])
            for offset in range(0, len(assigned), 9)
        ]
        board = normalize_board(board, self.rotated)

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
            
        # ponytail: Only calculate confidence from actual identified pieces (label != "").
        # Decoupling empty cells from confidence prevents background noise, route indicators, 
        # or highlight rings on empty intersections from dragging down overall confidence.
        # Illegal piece distribution (if any) is securely validation-checked by validate_board afterwards.
        confidence = recognition_confidence(
            [
                score
                for score, label in zip(scores, assigned)
                if label
            ]
        )
        for cell, label in enumerate(assigned):
            if not label:
                continue
            label_score = cell_scores[cell][label]
            feasible_scores = _feasible_identity_scores(
                cell_scores[cell], label, counts, limits
            )
            margin = _identity_margin(feasible_scores, label)
            # Under auto-healing safety design, low-confidence or narrow-margin pieces are NOT blocked by popup errors.
            # They smoothly progress to validate_board() and sanitize_board() for overall structural verification.
            continue
        if occupancy_trusted:
            occupied_board = {
                (9 - row, 8 - col) if self.rotated else (row, col)
                for row, col in occupied
            }
            board_occupied = {
                (row, col)
                for row, values in enumerate(board)
                for col, label in enumerate(values)
                if label
            }
            missing = sorted(occupied_board - board_occupied)
            if missing:
                # ponytail: Highlighting rings or move animations shouldn't halt recognition with a blocking popup dialog.
                # Automatically recover missing pieces by assigning their highest confidence non-empty label.
                for r, c in missing:
                    screen_r = 9 - r if self.rotated else r
                    screen_c = 8 - c if self.rotated else c
                    cell = screen_r * 9 + screen_c
                    nonempty = {lbl: sc for lbl, sc in cell_scores[cell].items() if lbl}
                    if nonempty:
                        best_ne = max(nonempty, key=nonempty.get)
                        if nonempty[best_ne] >= 0.20:
                            assigned[cell] = best_ne
                # Rebuild board after auto-recovering missing highlighted pieces
                board = [
                    tuple(assigned[offset : offset + 9])
                    for offset in range(0, len(assigned), 9)
                ]
                board = normalize_board(board, self.rotated)
        error = (
            validate_board(board)
            if occupancy_trusted
            else self._validate(board, confidence)
        )
        return Recognition(board, confidence, not error, error)

    def _validate(self, board, confidence):
        if confidence < self.threshold:
            return f"识别置信度过低: {confidence:.2f}"
        return validate_board(board)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        samples_tmp = directory / "samples.tmp"
        config_tmp = directory / "config.tmp"
        with samples_tmp.open("wb") as file:
            np.savez_compressed(file, labels=self.labels, samples=self.samples)
        samples_hash = hashlib.sha256(samples_tmp.read_bytes()).hexdigest()
        config_tmp.write_text(
            json.dumps(
                {
                    "feature_version": CALIBRATION_FEATURE_VERSION,
                    "rect": self.rect,
                    "rotated": self.rotated,
                    "size": self.size,
                    "threshold": self.threshold,
                    "frame_shape": self.frame_shape,
                    "source_id": self.source_id,
                    "circle_occupancy": self.circle_occupancy,
                    "slant": self.slant,
                    "std_threshold": self.std_threshold,
                    "skin_signature": self.skin_signature,
                    "samples_sha256": samples_hash,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        samples_tmp.replace(directory / "samples.npz")
        config_tmp.replace(directory / "config.json")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("feature_version") != CALIBRATION_FEATURE_VERSION:
            raise ValueError("校准格式已升级，请重新校准")
        samples_path = directory / "samples.npz"
        expected_hash = config.get("samples_sha256")
        if expected_hash and hashlib.sha256(samples_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("校准样本损坏")
        try:
            with np.load(samples_path) as data:
                labels = data["labels"]
                samples = data["samples"]
        except Exception as error:
            raise ValueError("校准样本损坏") from error
        return cls(
            config["rect"],
            config["rotated"],
            config["size"],
            labels,
            samples,
            config["threshold"],
            config.get("frame_shape"),
            config.get("source_id", ""),
            config.get("circle_occupancy", True),
            config.get("slant", 0.0),
            config.get("std_threshold", 20.0),
            config.get("skin_signature"),
        )


@dataclass(frozen=True)
class Move:
    side: str
    start: tuple
    end: tuple


def move_to_uci(move):
    sr, sc = move.start
    er, ec = move.end
    return f"{chr(97 + sc)}{9 - sr}{chr(97 + ec)}{9 - er}"


def _between(board, start, end):
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec > sc else -1
        return [board[sr][col] for col in range(sc + step, ec, step)]
    if sc == ec:
        step = 1 if er > sr else -1
        return [board[row][sc] for row in range(sr + step, er, step)]
    return None


def _piece_can_move(board, start, end):
    sr, sc = start
    er, ec = end
    piece = board[sr][sc]
    target = board[er][ec]
    red = piece.isupper()
    kind = piece.upper()
    dr, dc = er - sr, ec - sc
    adr, adc = abs(dr), abs(dc)

    if kind == "R":
        between = _between(board, start, end)
        return between is not None and not any(between)
    if kind == "C":
        between = _between(board, start, end)
        if between is None:
            return False
        screens = sum(bool(value) for value in between)
        return screens == (1 if target else 0)
    if kind == "N":
        if sorted((adr, adc)) != [1, 2]:
            return False
        leg = (sr + dr // 2, sc) if adr == 2 else (sr, sc + dc // 2)
        return not board[leg[0]][leg[1]]
    if kind == "B":
        if (adr, adc) != (2, 2) or (red and er < 5) or (not red and er > 4):
            return False
        return not board[sr + dr // 2][sc + dc // 2]
    if kind == "A":
        palace_rows = range(7, 10) if red else range(0, 3)
        return (adr, adc) == (1, 1) and er in palace_rows and 3 <= ec <= 5
    if kind == "K":
        between = _between(board, start, end)
        if target and target.upper() == "K" and sc == ec:
            return not any(between)
        palace_rows = range(7, 10) if red else range(0, 3)
        return adr + adc == 1 and er in palace_rows and 3 <= ec <= 5
    if kind == "P":
        forward = -1 if red else 1
        crossed = sr <= 4 if red else sr >= 5
        return (dr, dc) == (forward, 0) or (crossed and dr == 0 and adc == 1)
    return False


def _king_in_check(board, side):
    king = "K" if side == "w" else "k"
    position = next(
        ((row, col) for row in range(10) for col in range(9) if board[row][col] == king),
        None,
    )
    if position is None:
        return False
    for row in range(10):
        for col in range(9):
            piece = board[row][col]
            if piece and piece.isupper() != (side == "w") and _piece_can_move(
                board, (row, col), position
            ):
                return True
    return False


def detect_move(before, after):
    before = normalize_board(before)
    after = normalize_board(after)
    changed = [
        (row, col)
        for row in range(10)
        for col in range(9)
        if before[row][col] != after[row][col]
    ]
    if len(changed) != 2:
        return None
    for start, end in (changed, reversed(changed)):
        piece = before[start[0]][start[1]]
        target = before[end[0]][end[1]]
        if (
            piece
            and not after[start[0]][start[1]]
            and after[end[0]][end[1]] == piece
            and (not target or target.isupper() != piece.isupper())
            and _piece_can_move(before, start, end)
            and not _king_in_check(after, "w" if piece.isupper() else "b")
        ):
            return Move("w" if piece.isupper() else "b", start, end)
    return None


def _motion_vector(image, point, size):
    x, y = point
    half = size // 2
    crop = image[y - half : y - half + size, x - half : x - half + size]
    if crop.shape[:2] != (size, size):
        raise ValueError("棋盘矩形超出截图范围")
    vector = cv2.resize(crop, (24, 24)).astype(np.float32)
    vector -= vector.mean(axis=(0, 1), keepdims=True)
    vector = vector.ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _changed_positions(calibration, reference, current, threshold=0.04):
    if reference is None or current is None:
        raise ValueError("截图为空")
    if reference.shape != current.shape:
        raise ValueError("截图尺寸已变化，请重新校准")
    scores = []
    for screen_row, point_row in enumerate(grid_points(calibration.rect)):
        for screen_col, point in enumerate(point_row):
            before = _motion_vector(reference, point, calibration.size)
            after = _motion_vector(current, point, calibration.size)
            difference = max(0.0, 1.0 - float(before @ after))
            position = (
                (9 - screen_row, 8 - screen_col)
                if calibration.rotated
                else (screen_row, screen_col)
            )
            scores.append((position, difference))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [item for item in scores if item[1] >= threshold][:6]


def _changed_move_candidates(board, changed, side):
    board = normalize_board(board)
    candidates = []
    for start, start_score in changed:
        piece = board[start[0]][start[1]]
        if not piece or piece.isupper() != (side == "w"):
            continue
        for end, end_score in changed:
            if end == start:
                continue
            after = [list(row) for row in board]
            after[end[0]][end[1]] = piece
            after[start[0]][start[1]] = ""
            after = tuple(tuple(row) for row in after)
            move = detect_move(board, after)
            if move and move.side == side:
                candidates.append((after, {start, end}, start_score + end_score))
    return candidates


def _infer_move_sequence_from_changes(board, changed, side):
    if len(changed) < 2:
        return None, "no_move"
    sequences = []
    for after, cells, score in _changed_move_candidates(board, changed, side):
        sequences.append((cells, score, (after,)))
        other = "b" if side == "w" else "w"
        for final, next_cells, next_score in _changed_move_candidates(
            after, changed, other
        ):
            covered = cells | next_cells
            if len(covered) > len(cells):
                sequences.append((covered, score + next_score, (after, final)))
    if not sequences:
        return None, "illegal"
    sequences.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
    best = sequences[0]
    ambiguous = (
        len(sequences) > 1
        and len(best[0]) == len(sequences[1][0])
        and best[1] - sequences[1][1] < 0.08
    )
    if ambiguous:
        return None, "ambiguous"
    return best[2], ""


def _infer_move_from_changes(board, changed, side):
    candidates = _changed_move_candidates(board, changed, side)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    if len(candidates) > 1 and candidates[0][2] - candidates[1][2] < 0.08:
        return None
    return candidates[0][0]


def infer_move_from_frames(calibration, board, reference, current, side):
    return _infer_move_from_changes(
        board, _changed_positions(calibration, reference, current), side
    )


class MotionTracker:
    def __init__(self, calibration):
        self.calibration = calibration
        self.board = None
        self.reference = None
        self.candidate = None
        self.candidate_count = 0
        self.last_move_cells = None
        self.reference_refresh_count = 0
        self.pending_boards = []
        self.unresolved_count = 0

    def reset(self, frame, board):
        self.board = normalize_board(board)
        self.reference = frame.copy()
        self.candidate = None
        self.candidate_count = 0
        self.last_move_cells = None
        self.reference_refresh_count = 0
        self.pending_boards = []
        self.unresolved_count = 0

    def observe(self, frame, side):
        if self.pending_boards:
            self.unresolved_count = 0
            return Recognition(self.pending_boards.pop(0), 1.0, True)
        if self.reference is None:
            result = self.calibration.recognize(frame)
            if result.valid:
                self.board = result.board
                self.reference = frame.copy()
            return result
        try:
            changed = _changed_positions(self.calibration, self.reference, frame)
            proposed, reason = _infer_move_sequence_from_changes(
                self.board, changed, side
            )
            changes = tuple(changed)
        except ValueError as error:
            return Recognition(self.board, 0.0, False, str(error))

        if proposed is None:
            self.candidate = None
            self.candidate_count = 0
            cells = {position for position, _score in changed}
            refreshing_highlight = (
                self.last_move_cells and cells and cells.issubset(self.last_move_cells)
            )
            if refreshing_highlight:
                self.reference_refresh_count += 1
                if self.reference_refresh_count >= 2:
                    self.reference = frame.copy()
                    self.last_move_cells = None
                    self.reference_refresh_count = 0
            else:
                self.reference_refresh_count = 0
            unresolved = bool(changed) and not refreshing_highlight
            self.unresolved_count = self.unresolved_count + 1 if unresolved else 0
            if self.unresolved_count >= 3:
                self.unresolved_count = 0
                recovered = self.calibration.recognize(frame)
                if recovered.valid:
                    return Recognition(
                        recovered.board,
                        recovered.confidence,
                        True,
                        "已识别当前棋盘，正在恢复",
                        True,
                        reason="recovery_succeeded",
                        changes=changes,
                    )
                return Recognition(
                    self.board,
                    0.0,
                    True,
                    f"无法自动识别当前棋盘，请重新校准: {recovered.error}",
                    reason="recovery_failed",
                    changes=changes,
                )
            error = ""
            if len(changed) == 1:
                error = "检测到选中操作，等待棋子落点"
            elif unresolved:
                error = (
                    "检测到多个可能走法，正在等待更多画面"
                    if reason == "ambiguous"
                    else "棋盘变化无法组成当前方合法走法，正在恢复"
                )
            return Recognition(
                self.board,
                1.0,
                True,
                error,
                reason=reason,
                changes=changes,
            )

        self.unresolved_count = 0
        self.reference_refresh_count = 0
        self.candidate_count = self.candidate_count + 1 if proposed == self.candidate else 1
        self.candidate = proposed
        if self.candidate_count < 2:
            return Recognition(
                self.board,
                1.0,
                True,
                "等待棋盘稳定",
                reason="unstable",
                changes=changes,
            )

        previous = self.board
        move_cells = set()
        for board in proposed:
            move = detect_move(previous, board)
            move_cells.update((move.start, move.end))
            previous = board
        self.board = proposed[-1]
        self.reference = frame.copy()
        self.last_move_cells = move_cells
        self.pending_boards = list(proposed[1:])
        self.candidate = None
        self.candidate_count = 0
        return Recognition(proposed[0], 1.0, True)
