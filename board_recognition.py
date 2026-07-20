import json
import hashlib
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


def detect_board_rect(image):
    if image is None or image.ndim != 3:
        raise ValueError("截图为空")
    short_side = min(image.shape[:2])
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=short_side / 22,
        param1=100,
        param2=34,
        minRadius=max(8, int(short_side * 0.018)),
        maxRadius=max(16, int(short_side * 0.065)),
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
    if not pairs:
        raise ValueError("未找到上下两排各9个等距棋子")
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


def recognition_confidence(scores):
    return float(np.percentile(scores, 5)) if scores else 0.0


def normalize_board(board, rotated=False):
    normalized = tuple(tuple(row) for row in board)
    if len(normalized) != 10 or any(len(row) != 9 for row in normalized):
        raise ValueError("棋盘必须是10行9列")
    if rotated:
        return tuple(tuple(reversed(row)) for row in reversed(normalized))
    return normalized


def board_to_fen(board, active="w"):
    board = normalize_board(board)
    if active not in {"w", "b"}:
        raise ValueError("行棋方必须是w或b")
    rows = []
    for row in board:
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
    board = normalize_board(board)
    pieces = [piece for row in board for piece in row if piece]
    if pieces.count("K") != 1 or pieces.count("k") != 1:
        return "红帅或黑将数量错误"
    limits = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}
    names = {"K": "帅/将", "A": "仕/士", "B": "相/象", "N": "马", "R": "车", "C": "炮", "P": "兵/卒"}
    for piece, limit in limits.items():
        if pieces.count(piece) > limit or pieces.count(piece.lower()) > limit:
            return f"棋子数量超过上限: {names[piece]}"
    if not 2 <= len(pieces) <= 32:
        return f"棋子总数错误: {len(pieces)}"
    red_king = next((position for position in ((row, col) for row in range(10) for col in range(9)) if board[position[0]][position[1]] == "K"), None)
    black_king = next((position for position in ((row, col) for row in range(10) for col in range(9)) if board[position[0]][position[1]] == "k"), None)
    if red_king[0] not in range(7, 10) or red_king[1] not in range(3, 6):
        return "红帅不在九宫内"
    if black_king[0] not in range(0, 3) or black_king[1] not in range(3, 6):
        return "黑将不在九宫内"
    if red_king[1] == black_king[1] and not any(
        board[row][red_king[1]] for row in range(black_king[0] + 1, red_king[0])
    ):
        return "将帅不能照面"

    # Restricted piece legal coordinates check
    legal_b = {(0,2),(0,6),(2,0),(2,4),(2,8),(4,2),(4,6)}
    legal_B = {(5,2),(5,6),(7,0),(7,4),(7,8),(9,2),(9,6)}
    legal_a = {(0,3),(0,5),(1,4),(2,3),(2,5)}
    legal_A = {(7,3),(7,5),(8,4),(9,3),(9,5)}

    for r in range(10):
        for c in range(9):
            piece = board[r][c]
            if piece == 'b' and (r, c) not in legal_b:
                return f"黑象在非法坐标: {(r, c)}"
            if piece == 'B' and (r, c) not in legal_B:
                return f"红相在非法坐标: {(r, c)}"
            if piece == 'a' and (r, c) not in legal_a:
                return f"黑士在非法坐标: {(r, c)}"
            if piece == 'A' and (r, c) not in legal_A:
                return f"红仕在非法坐标: {(r, c)}"

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


class Calibration:
    def __init__(
        self, rect, rotated, size, labels, samples, threshold=0.45,
        frame_shape=None, source_id="", circle_occupancy=False, slant=0.0,
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
        labels, samples = [], []
        for row, point_row in zip(screen_board, points):
            for label, point in zip(row, point_row):
                labels.append(label)
                samples.append(cls._vector(image, point, size))
        expected_occupied = {
            (row, col)
            for row, values in enumerate(screen_board)
            for col, label in enumerate(values)
            if label
        }
        calibration = cls(
            rect, rotated, size, labels, samples,
            frame_shape=image.shape[:2], source_id=source_id,
            circle_occupancy=(
                cls._occupied_cells(image, rect, size) == expected_occupied
            ),
            slant=slant,
        )
        result = calibration.recognize(image)
        if not result.valid or result.board != STANDARD_BOARD:
            raise ValueError(f"标准开局校准失败: {result.error or '棋盘不一致'}")
        return calibration

    @staticmethod
    def _vector(image, point, size):
        x, y = point
        half = size // 2
        top, bottom = y - half, y - half + size
        left, right = x - half, x - half + size
        if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
            raise ValueError("棋盘矩形超出截图范围")
        crop = cv2.resize(image[top:bottom, left:right], (40, 40))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Heal outer ring to suppress selection/highlight overlays (e.g. selection rings, banners)
        yy, xx = np.ogrid[:40, :40]
        dist_sq = (xx - 19.5) ** 2 + (yy - 19.5) ** 2
        inner_mask = (dist_sq >= 11 ** 2) & (dist_sq <= 13 ** 2)
        if np.any(inner_mask):
            inner_mean = gray[inner_mask].mean()
            # Replace the outer ring (radius > 13.5) with the inner mean
            outer_mask = dist_sq > 13.5 ** 2
            gray[outer_mask] = inner_mean
            
        vector = gray[dist_sq <= 18 ** 2]
        
        # ponytail: If the cell has low contrast/no features, return a zero vector
        # to prevent standardizing background noise into a full-scale feature vector.
        if np.std(vector) < 3.0:
            return np.zeros_like(vector)
            
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def _occupied_cells(image, rect, size):
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
        minimum_radius = max(8, int(size * 0.34))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12, size * 0.6),
            param1=100,
            param2=28,
            minRadius=minimum_radius,
            maxRadius=max(minimum_radius + 2, int(size * 0.7)),
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

    def recognize(self, image):
        if image is None:
            return Recognition(tuple(), 0.0, False, "截图为空")
        if self.frame_shape and tuple(image.shape[:2]) != self.frame_shape:
            return Recognition(tuple(), 0.0, False, "截图尺寸已变化，请重新校准")
        labels = tuple(dict.fromkeys(str(label) for label in self.labels))
        label_indexes = {
            label: np.flatnonzero(self.labels == label) for label in labels
        }
        cell_scores = []
        for point_row in grid_points(self.rect, self.slant):
            for point in point_row:
                try:
                    vector = self._vector(image, point, self.size)
                except ValueError as error:
                    return Recognition(tuple(), 0.0, False, str(error))
                similarities = self.samples @ vector

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
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    hue, saturation, value = cv2.split(hsv)
                    red_mask = (
                        ((hue < 12) | (hue > 170))
                        & (saturation > 90)
                        & (value > 70)
                    )
                    if np.count_nonzero(red_mask) >= max(3, int(crop.size * 0.002)):
                        has_red = True

                is_empty = np.all(vector == 0)
                scores = {}
                for label, indexes in label_indexes.items():
                    if is_empty:
                        scores[label] = 1.0 if not label else 0.0
                    else:
                        if label:
                            # Side constraint
                            if label.isupper() and not has_red:
                                scores[label] = 0.0
                            elif label.islower() and has_red:
                                scores[label] = 0.0
                            else:
                                scores[label] = float(np.max(similarities[indexes]))
                        else:
                            scores[label] = float(np.max(similarities[indexes]))

                # ponytail: If the best match score at the center is low (< 0.70) and not empty,
                # search a small neighborhood (dx in [-3..3], dy in [-3..3]) to align shifted/highlighted pieces.
                best_label = max(scores, key=scores.get)
                if not is_empty and scores[best_label] < 0.70:
                    shifts = []
                    for r_dist in range(1, 4):
                        for dx in range(-r_dist, r_dist + 1):
                            for dy in range(-r_dist, r_dist + 1):
                                if max(abs(dx), abs(dy)) == r_dist:
                                    shifts.append((dx, dy))
                    for dx, dy in shifts:
                        shifted_pt = (point[0] + dx, point[1] + dy)
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
                                if label.isupper() and not has_red:
                                    s_scores[label] = 0.0
                                elif label.islower() and has_red:
                                    s_scores[label] = 0.0
                                else:
                                    s_scores[label] = float(np.max(s_similarities[indexes]))
                            else:
                                s_scores[label] = float(np.max(s_similarities[indexes]))
                        s_best_label = max(s_scores, key=s_scores.get)
                        if s_best_label and s_scores[s_best_label] > scores[best_label]:
                            scores = s_scores
                            best_label = s_best_label
                            if scores[best_label] >= 0.85:
                                break

                cell_scores.append(scores)

        limits = {
            label: int(np.count_nonzero(self.labels == label))
            for label in labels
            if label
        }
        occupied = self._occupied_cells(image, self.rect, self.size)
        # ponytail: JJ圆形皮肤提供24+强圆证据；新皮肤若达到该值再改用填充剖面。
        if len(occupied) >= 24:
            self.circle_occupancy = True
        occupancy_trusted = bool(
            self.circle_occupancy and 2 <= len(occupied) <= 32
        )
        assigned = []
        for cell, values in enumerate(cell_scores):
            row, col = divmod(cell, 9)
            if occupancy_trusted and (row, col) in occupied:
                nonempty = {
                    label: score for label, score in values.items() if label
                }
                label = max(nonempty, key=nonempty.get) if nonempty else ""
            else:
                label = max(values, key=values.get)
            assigned.append(label)
        if occupancy_trusted:
            # ponytail: If a piece matches its templates very strongly, keep it even if
            # HoughCircles misses it due to selection rings or move highlights.
            # Now with the empty filter checking standard deviation, we can safely lower
            # this threshold back to protect highlighted pieces without empty cell false positives.
            assigned = [
                label if (divmod(cell, 9) in occupied or (label and cell_scores[cell][label] >= max(0.55, 1.0001 if self.threshold + 0.10 >= 1.0 else self.threshold + 0.10))) else ""
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
                    alternatives = [
                        candidate
                        for candidate in labels
                        if candidate != label
                        and (
                            not candidate
                            or counts[candidate] < limits[candidate]
                        )
                    ]
                    replacement = max(
                        alternatives,
                        key=cell_scores[cell].get,
                    )
                    penalty = (
                        cell_scores[cell][label]
                        - cell_scores[cell][replacement]
                    )
                    replacements.append((penalty, cell, label, replacement))
            _penalty, cell, old, new = min(replacements)
            assigned[cell] = new
            counts[old] -= 1
            if new:
                counts[new] += 1

        scores = [
            cell_scores[cell][label]
            for cell, label in enumerate(assigned)
        ]

        board = [
            tuple(assigned[offset : offset + 9])
            for offset in range(0, len(assigned), 9)
        ]
        board = normalize_board(board, self.rotated)

        # Clear restricted pieces on illegal positions
        legal_b = {(0,2),(0,6),(2,0),(2,4),(2,8),(4,2),(4,6)}
        legal_B = {(5,2),(5,6),(7,0),(7,4),(7,8),(9,2),(9,6)}
        legal_a = {(0,3),(0,5),(1,4),(2,3),(2,5)}
        legal_A = {(7,3),(7,5),(8,4),(9,3),(9,5)}
        mutable_board = [list(row) for row in board]
        corrected = False
        for r in range(10):
            for c in range(9):
                p = mutable_board[r][c]
                if p == 'b' and (r, c) not in legal_b:
                    mutable_board[r][c] = ""
                    corrected = True
                elif p == 'B' and (r, c) not in legal_B:
                    mutable_board[r][c] = ""
                    corrected = True
                elif p == 'a' and (r, c) not in legal_a:
                    mutable_board[r][c] = ""
                    corrected = True
                elif p == 'A' and (r, c) not in legal_A:
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
        if occupancy_trusted:
            missing = sorted(
                occupied
                - {
                    divmod(cell, 9)
                    for cell, label in enumerate(assigned)
                    if label
                }
            )
            if missing:
                return Recognition(
                    board,
                    confidence,
                    False,
                    f"检测到棋子但无法确认身份: {missing}",
                )
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
                    "rect": self.rect,
                    "rotated": self.rotated,
                    "size": self.size,
                    "threshold": self.threshold,
                    "frame_shape": self.frame_shape,
                    "source_id": self.source_id,
                    "circle_occupancy": self.circle_occupancy,
                    "slant": self.slant,
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
