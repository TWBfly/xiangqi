import os
import queue
import threading
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from board_recognition import (
    Calibration,
    MotionTracker,
    STANDARD_BOARD,
    board_to_fen,
    detect_board_rect,
    detect_move,
    infer_player_from_colors,
    grid_points,
    is_legal_xiangqi_move,
)
from capture import (
    adb_devices,
    adb_frame,
    enable_dpi_awareness,
    find_adb,
    list_emulator_windows,
    window_frame,
)
from pikafish_engine import PikafishEngine
from game_state import GameState


DEFAULT_SEARCH_SECONDS = 20
POLL_INTERVAL_SECONDS = 0.2
APP_VERSION = "2026.07.23.2"


def format_analysis_result(move, score, text):
    status_suffix = ""
    try:
        val_str = str(score).split()[0]
        val = float(val_str)
        if val <= -4.0:
            status_suffix = " (大劣局面/黑胜势)"
        elif val >= 4.0:
            status_suffix = " (大优局面)"
    except (ValueError, IndexError):
        if "Mate" in str(score):
            status_suffix = " (将被绝杀/死棋)" if "-" in str(score) else " (绝杀)"
    return text, f"{move} · {score}{status_suffix}"


def save_diagnostic_frames(directory, reference, current):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = {
        "diagnostic_reference.png": reference,
        "diagnostic_current.png": current,
    }
    for name, image in images.items():
        if image is None or not cv2.imwrite(str(directory / name), image):
            raise OSError(f"无法保存诊断截图: {name}")


def save_analysis_snapshot(directory, frame):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "analysis_snapshot.png"
    if frame is None or not cv2.imwrite(str(path), frame):
        raise OSError(f"无法保存分析快照: {path}")


def load_diagnostic_images(directory, max_size=(400, 320)):
    directory = Path(directory)
    items = []
    for title, name in (
        ("最近一次同步分析快照", "analysis_snapshot.png"),
    ):
        path = directory / name
        item = {
            "title": title,
            "path": path,
            "image": None,
            "modified": "",
            "error": "",
        }
        if not path.exists():
            item["error"] = "图片不存在"
        else:
            try:
                with Image.open(path) as source:
                    preview = source.copy()
                preview.thumbnail(max_size)
                item["image"] = preview
                item["modified"] = datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError) as error:
                item["error"] = f"无法读取图片: {error}"
        items.append(item)
    return items


def window_source_id(title, hwnd, title_count):
    return f"window:{title}" if title_count == 1 else f"window:{title}:{hwnd}"


def create_calibration(frame, rotated, source_id, choose_manual_rect):
    try:
        rect = detect_board_rect(frame)
        return Calibration.create(frame, rect, rotated, source_id), "自动"
    except ValueError as automatic_error:
        rect = choose_manual_rect(str(automatic_error))
        if not rect:
            return None, ""
        return Calibration.create(frame, rect, rotated, source_id), "手动"


def parse_movetime_seconds(value):
    try:
        seconds = int(value)
    except ValueError as error:
        raise ValueError("搜索秒数必须是1到60的整数") from error
    if not 1 <= seconds <= 60:
        raise ValueError("搜索秒数必须是1到60的整数")
    return seconds * 1000


class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"象棋辅助 {APP_VERSION}")
        self.root.geometry("360x700")
        self.root.resizable(False, True)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.sources = {}
        self.results = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.engine = None
        self.engine_lock = threading.Lock()
        self.closing = False
        self.calibration = None
        self.synced_position = None
        self.pending_recovery_board = None
        self.config_dir = Path(
            os.environ.get("APPDATA", Path.home() / ".xiangqi-assistant")
        ) / "XiangqiAssistant"

        self.source_var = tk.StringVar()
        self.player_var = tk.StringVar(value="红方")
        self.side_to_move_var = tk.StringVar(value="红方")
        self.movetime_var = tk.StringVar(value=str(DEFAULT_SEARCH_SECONDS))
        self.move_var = tk.StringVar(value="--")
        self.source_var = tk.StringVar()
        self.player_var = tk.StringVar(value="红方")
        self.side_to_move_var = tk.StringVar(value="红方")
        self.movetime_var = tk.StringVar(value=str(DEFAULT_SEARCH_SECONDS))
        self.move_var = tk.StringVar(value="--")
        self.status_var = tk.StringVar(value="AI已就绪，选择采集源即可同步分析")
        self.preview_photo = None

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="采集源").grid(row=0, column=0, sticky="w")
        self.source_box = ttk.Combobox(
            frame, textvariable=self.source_var, state="readonly", width=29
        )
        self.source_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="我执").grid(row=1, column=0, sticky="w", pady=8)
        self.player_box = ttk.Combobox(
            frame,
            textvariable=self.player_var,
            state="readonly",
            values=("红方", "黑方"),
            width=8,
        )
        self.player_box.grid(row=1, column=1, sticky="w", padx=(8, 0))
        self.player_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.side_to_move_var.set(self.player_var.get()),
        )
        ttk.Button(frame, text="刷新设备", command=self.refresh_sources).grid(
            row=1, column=2, sticky="e"
        )

        ttk.Label(frame, text="本次分析方").grid(row=2, column=0, sticky="w")
        self.side_to_move_box = ttk.Combobox(
            frame,
            textvariable=self.side_to_move_var,
            state="disabled",
            values=("红方", "黑方"),
            width=8,
        )
        self.side_to_move_box.grid(row=2, column=1, sticky="w", padx=(8, 0))
        search_frame = ttk.Frame(frame)
        search_frame.grid(row=2, column=2, sticky="e")
        ttk.Label(search_frame, text="最长秒").pack(side="left")
        ttk.Label(search_frame, text="").pack(side="left")  # Spacer
        ttk.Spinbox(
            search_frame, from_=1, to=60, textvariable=self.movetime_var, width=4
        ).pack(side="left", padx=(4, 0))

        self.sync_button = ttk.Button(
            frame, text="一键同步分析", command=self.sync_current_board
        )
        self.sync_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )
        self.start_button = ttk.Button(
            frame, text="停止分析", command=self.stop, state="disabled"
        )
        self.start_button.grid(row=3, column=2, sticky="ew", padx=(8, 0), pady=(8, 4))

        ttk.Button(frame, text="重新校准/定位", command=self.calibrate).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4)
        )
        ttk.Button(
            frame, text="查看诊断", command=self.show_diagnostics
        ).grid(row=4, column=2, sticky="ew", padx=(8, 0), pady=(0, 4))

        ttk.Label(frame, textvariable=self.move_var, anchor="center", font=("Microsoft YaHei", 28, "bold")).grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=4
        )
        self.status_label = ttk.Label(frame, textvariable=self.status_var, anchor="center", wraplength=480, justify="center", cursor="hand2")
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.status_label.bind("<Button-1>", lambda e: messagebox.showinfo("状态/错误详情", self.status_var.get()))
        self.preview_label = ttk.Label(frame, anchor="center")
        self.preview_label.grid(row=7, column=0, columnspan=3, pady=(6, 0))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        try:
            self.calibration = Calibration.load(self.config_dir)
            self.calibration.align_grid = True
            player_text = "黑方" if self.calibration.rotated else "红方"
            self.player_var.set(player_text)
            self.side_to_move_var.set(player_text)
            self.status_var.set("AI已就绪，选择采集源即可同步分析")
        except (OSError, ValueError, KeyError):
            pass
        self.refresh_sources()
        self.root.after(100, self.poll_results)

    def refresh_sources(self):
        self.sources.clear()
        adb = find_adb()
        if adb:
            try:
                for serial in adb_devices(adb):
                    self.sources[f"ADB · {serial}"] = ("adb", adb, serial, f"adb:{serial}")
            except (OSError, RuntimeError):
                pass
        windows = list_emulator_windows()
        title_counts = Counter(title for _, title in windows)
        for hwnd, title in windows:
            self.sources[f"窗口 · {title} · {hwnd}"] = (
                "window", hwnd, window_source_id(title, hwnd, title_counts[title])
            )
        values = list(self.sources)
        self.source_box["values"] = values
        if values:
            if self.source_var.get() not in self.sources:
                self.source_var.set(values[0])
            self.status_var.set("设备已就绪" if self.calibration else "设备已就绪，请校准")
        else:
            self.source_var.set("")
            self.status_var.set("未找到模拟器，请启动模拟器后刷新")

    def capture(self):
        source = self.sources.get(self.source_var.get())
        if not source:
            raise RuntimeError("未选择可用模拟器")
        return adb_frame(source[1], source[2]) if source[0] == "adb" else window_frame(source[1])

    def source_id(self):
        source = self.sources.get(self.source_var.get())
        return source[-1] if source else ""

    def calibrate(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("象棋辅助", "请先停止辅助")
            return
        try:
            frame = self.capture()
            rotated = self.player_var.get() == "黑方"

            def choose_manual(error):
                self.status_var.set(f"自动校准失败，已进入手动拖框: {error}")
                return self.choose_board_rect(frame, error)

            calibration, mode = create_calibration(
                frame, rotated, self.source_id(), choose_manual
            )
            if not calibration:
                return
            detected_player = infer_player_from_colors(frame, calibration.rect)
            automatic_player = detected_player is not None
            if automatic_player:
                rotated = detected_player == "b"
                if calibration.rotated != rotated:
                    calibration = Calibration.create(
                        frame, calibration.rect, rotated, self.source_id()
                    )
                player_text = "红方" if detected_player == "w" else "黑方"
                self.player_var.set(player_text)
                self.side_to_move_var.set(player_text)
            else:
                detected_player = "b" if calibration.rotated else "w"
            calibration.save(self.config_dir)
            self.calibration = calibration
            self.calibration.align_grid = True
            self.move_var.set("--")
            player_text = "红方" if detected_player == "w" else "黑方"
            suffix = "" if automatic_player else "（使用人工选择）"
            self.status_var.set(
                f"{mode}校准成功，已识别我执{player_text}{suffix}"
            )
        except Exception as error:
            messagebox.showerror("校准失败", str(error))
            self.status_var.set(f"校准失败: {error}")

    def choose_board_rect(self, frame, automatic_error=""):
        result = []
        dialog = tk.Toplevel(self.root)
        dialog.title(
            f"自动校准失败（{automatic_error}），请拖框：左上棋子中心 → 右下棋子中心"
        )
        dialog.attributes("-topmost", True)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((800, 720))
        shown = ImageTk.PhotoImage(image)
        canvas = tk.Canvas(dialog, width=image.width, height=image.height, cursor="cross")
        canvas.pack()
        canvas.create_image(0, 0, image=shown, anchor="nw")
        canvas.image = shown
        start = {}
        rectangle = {"id": None}

        def press(event):
            start.update(x=event.x, y=event.y)
            if rectangle["id"]:
                canvas.delete(rectangle["id"])
            rectangle["id"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#00ff00", width=2
            )

        def drag(event):
            if rectangle["id"]:
                canvas.coords(rectangle["id"], start["x"], start["y"], event.x, event.y)

        def release(event):
            x1, x2 = sorted((start["x"], event.x))
            y1, y2 = sorted((start["y"], event.y))
            if x2 - x1 < 100 or y2 - y1 < 100:
                return
            sx, sy = frame.shape[1] / image.width, frame.shape[0] / image.height
            result.append(tuple(round(value) for value in (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
            dialog.destroy()

        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result[0] if result else None

    def toggle(self):
        if self.worker and self.worker.is_alive():
            self.stop()
            return
        if not self.calibration:
            messagebox.showinfo("象棋辅助", "请先在标准开局校准棋盘")
            return
        if self.calibration.rotated != (self.player_var.get() == "黑方"):
            messagebox.showinfo("象棋辅助", "执棋方已改变，请重新校准棋盘")
            return
        if self.calibration.source_id and self.calibration.source_id != self.source_id():
            messagebox.showinfo("象棋辅助", "模拟器采集源已改变，请重新校准棋盘")
            return
        if self.source_var.get() not in self.sources:
            messagebox.showinfo("象棋辅助", "请选择可用模拟器")
            return
        try:
            movetime = parse_movetime_seconds(self.movetime_var.get())
            player = "w" if self.player_var.get() == "红方" else "b"
            if self.synced_position:
                frame = self.capture()
                board, side_to_move, fen = self.synced_position
                state = GameState.start(board, side_to_move, player, fen)
                self.synced_position = None
            else:
                # Try to capture and recognize with a retry loop to wait out animations
                max_retries = 5
                recognized = None
                for attempt in range(max_retries):
                    frame = self.capture()
                    recognized = self.calibration.recognize(frame)
                    if recognized.valid:
                        break
                    if attempt < max_retries - 1:
                        self.status_var.set(f"置信度低，等待动画结束中 ({attempt + 1}/{max_retries})...")
                        if hasattr(self, "root") and hasattr(self.root, "update"):
                            self.root.update()
                        import time
                        time.sleep(0.15)
                if not recognized.valid:
                    raise ValueError(
                        f"无法识别当前棋盘，请重新校准: {recognized.error}"
                    )
                side_to_move = (
                    "w"
                    if recognized.board == STANDARD_BOARD
                    else ("w" if self.side_to_move_var.get() == "红方" else "b")
                )
                state = GameState.start(recognized.board, side_to_move, player)
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("无法开始辅助", str(error))
            self.status_var.set(str(error))
            return
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self.run_loop,
            args=(player, state, frame, movetime),
            daemon=True,
        )
        self.worker.start()
        self.start_button.configure(text="停止")
        self.player_box.configure(state="disabled")
        self.side_to_move_box.configure(state="disabled")
        self.status_var.set("正在识别棋盘")

    def sync_current_board(self):
        if self.worker and self.worker.is_alive():
            self.status_var.set("正在分析，请稍候或点击“停止分析”")
            return
        try:
            if not self.calibration:
                raise ValueError("请先在标准开局校准棋盘")
            player = "w" if self.player_var.get() == "红方" else "b"
            if self.calibration.rotated != (player == "b"):
                raise ValueError("执棋方已改变，请重新校准棋盘")
            if (
                self.calibration.source_id
                and self.calibration.source_id != self.source_id()
            ):
                raise ValueError("模拟器采集源已改变，请重新校准棋盘")
            if self.source_var.get() not in self.sources:
                raise ValueError("请选择可用模拟器")
            movetime = parse_movetime_seconds(self.movetime_var.get())
            self.status_var.set("正在截取当前棋盘")
            # Try to capture and recognize with a multi-frame best-confidence loop to wait out animations
            max_retries = 8
            best_recognized = None
            best_frame = None
            best_conf = -1.0

            for attempt in range(max_retries):
                frame = self.capture()
                self.move_var.set("--")
                self.status_var.set("正在识别当前棋盘")
                recognized = self.calibration.recognize(frame)
                
                if recognized.confidence > best_conf:
                    best_conf = recognized.confidence
                    best_recognized = recognized
                    best_frame = frame
                    
                if recognized.valid:
                    best_recognized = recognized
                    best_frame = frame
                    break
                if attempt < max_retries - 1:
                    self.status_var.set(f"置信度低，等待动画结束中 ({attempt + 1}/{max_retries})...")
                    if hasattr(self, "root") and hasattr(self.root, "update"):
                        self.root.update()
                    import time
                    time.sleep(0.18)

            recognized = best_recognized
            frame = best_frame
            if frame is not None:
                save_analysis_snapshot(self.config_dir, frame)
            if not recognized or not recognized.valid:
                error_msg = recognized.error if recognized else "未截取到有效图像"
                raise ValueError(
                    f"无法识别当前棋盘: {error_msg}。"
                    "请等待动画结束后重试；"
                    "仅在分辨率、棋盘皮肤或执棋方向改变时重新校准"
                )
            position = board_to_fen(recognized.board, player)
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("无法同步并分析", str(error))
            self.status_var.set(str(error))
            return
        self.side_to_move_var.set("红方" if player == "w" else "黑方")
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self.analyze_snapshot,
            args=(recognized.board, position, movetime, player),
            daemon=True,
        )
        self.worker.start()
        self.sync_button.configure(state="disabled")
        self.start_button.configure(state="normal")
        self.player_box.configure(state="disabled")
        self.status_var.set("已保存快照，正在分析")

    def open_diagnostic_directory(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            messagebox.showinfo("诊断目录", str(self.config_dir))
            return
        try:
            startfile(str(self.config_dir))
        except OSError as error:
            messagebox.showerror(
                "无法打开诊断目录", f"{error}\n{self.config_dir}"
            )

    def show_diagnostics(self):
        window = tk.Toplevel(self.root)
        window.title("诊断截图")
        window.geometry("900x500")
        window.attributes("-topmost", True)
        window._diagnostic_images = []
        items = load_diagnostic_images(self.config_dir)

        if not any(item["path"].exists() for item in items):
            ttk.Label(
                window,
                text=(
                    "尚无分析快照\n"
                    "点击“同步并分析”后，程序会保存实际参与分析的画面。\n"
                    f"保存目录：{self.config_dir}"
                ),
                anchor="center",
                justify="center",
            ).pack(fill="both", expand=True, padx=20, pady=20)
        else:
            body = ttk.Frame(window, padding=12)
            body.pack(fill="both", expand=True)
            for column, item in enumerate(items):
                panel = ttk.Frame(body)
                panel.grid(row=0, column=column, sticky="nsew", padx=6)
                ttk.Label(panel, text=item["title"]).pack()
                if item["image"] is not None:
                    photo = ImageTk.PhotoImage(item["image"])
                    window._diagnostic_images.append(photo)
                    ttk.Label(panel, image=photo).pack(pady=8)
                else:
                    ttk.Label(panel, text=item["error"]).pack(pady=40)
                ttk.Label(
                    panel,
                    text=(
                        f"{item['path'].name}\n快照时间：{item['modified']}\n"
                        f"{item['path']}"
                    ),
                    wraplength=400,
                    justify="center",
                ).pack()
                body.columnconfigure(column, weight=1)

        ttk.Button(
            window,
            text="打开诊断目录",
            command=self.open_diagnostic_directory,
        ).pack(pady=(0, 12))

    def stop(self):
        self.stop_event.set()
        with self.engine_lock:
            engine = self.engine
        if engine:
            engine.stop_search()
        self.start_button.configure(state="disabled")
        self.status_var.set("正在停止分析")

    def analyze_snapshot(self, board, position, movetime, side=None):
        try:
            if side is None:
                fields = position.split()
                side = (
                    fields[3]
                    if fields[:2] == ["position", "fen"]
                    else fields[1]
                )
            move, score, text = self.best_advice(
                board, position, side, movetime
            )
            if self.stop_event.is_set():
                return
            display_text, status_text = format_analysis_result(move, score, text)
            self.results.put(("move", display_text, status_text))
            
            # Generate the board preview with padding so edge pieces are completely visible
            img_path = self.config_dir / "analysis_snapshot.png"
            if img_path.exists():
                import cv2
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    left, top, right, bottom = self.calibration.rect
                    cell_w = (right - left) / 8.0
                    cell_h = (bottom - top) / 9.0
                    pad = int(max(32, max(cell_w, cell_h) * 0.65))

                    crop_left = max(0, left - pad)
                    crop_top = max(0, top - pad)
                    crop_right = min(frame.shape[1], right + pad)
                    crop_bottom = min(frame.shape[0], bottom + pad)

                    board_crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()
                    coords = PikafishEngine.uci_to_coords(move)
                    if coords:
                        sc, sr, ec, er = coords
                        if is_legal_xiangqi_move(board, sr, sc, er, ec, side):
                            points = grid_points(self.calibration.rect, self.calibration.slant)
                            x1 = points[sr][sc][0] - crop_left
                            y1 = points[sr][sc][1] - crop_top
                            x2 = points[er][ec][0] - crop_left
                            y2 = points[er][ec][1] - crop_top
                            cv2.arrowedLine(board_crop, (x1, y1), (x2, y2), (0, 0, 255), 4, tipLength=0.15)
                    self.results.put(("board_preview", board_crop))
        except Exception as error:
            if not self.stop_event.is_set():
                self.results.put(("error", str(error)))
        finally:
            self.results.put(("stopped",))

    def run_loop(
        self,
        player,
        state=None,
        initial_frame=None,
        movetime=DEFAULT_SEARCH_SECONDS * 1000,
    ):
        motion = MotionTracker(self.calibration)
        if state is not None and initial_frame is not None:
            motion.reset(initial_frame, state.board)
        if state is not None:
            self.results.put(("turn", state.side_to_move))
        try:
            while not self.stop_event.is_set():
                frame = self.capture()
                side_to_move = state.side_to_move if state else "w"
                result = motion.observe(frame, side_to_move)
                diagnostic_error = ""
                if result.reason in {"recovery_succeeded", "recovery_failed"}:
                    try:
                        save_diagnostic_frames(
                            self.config_dir, motion.reference, frame
                        )
                    except OSError as error:
                        diagnostic_error = f"；{error}"
                if result.changes:
                    self.results.put(("clear_move", "检测到棋盘变化，正在确认局面"))
                if not result.valid:
                    self.results.put(("status", result.error + diagnostic_error))
                    self.stop_event.wait(POLL_INTERVAL_SECONDS)
                    continue
                if state is None:
                    state = GameState.start(result.board, "w", player)
                    self.results.put(("turn", state.side_to_move))
                elif result.recovery:
                    if result.board == state.board:
                        motion.reset(frame, state.board)
                        self.results.put(("status", "当前局面已确认"))
                    elif state.apply_board(result.board):
                        motion.reset(frame, state.board)
                        self.results.put(("turn", state.side_to_move))
                        self.results.put(("status", "已自动补录漏掉的一步"))
                    elif state.restore_snapshot(result.board):
                        motion.reset(frame, state.board)
                        self.results.put(("turn", state.side_to_move))
                        self.results.put(("status", "已恢复到历史局面"))
                    else:
                        self.results.put(("needs_sync", result.board))
                        self.stop_event.set()
                        continue
                elif result.board != state.board:
                    if state.apply_board(result.board):
                        self.results.put(("turn", state.side_to_move))
                        if state.side_to_move != player:
                            self.results.put(("clear_move", "等待对手落子"))
                    else:
                        self.results.put(("status", "局面与当前回合不一致，正在恢复"))

                if state.needs_analysis():
                    try:
                        move, score, text = self.best_advice(
                            state.board,
                            state.position_command(),
                            state.side_to_move,
                            movetime,
                        )
                        state.mark_analyzed()
                        display_text, status_text = format_analysis_result(move, score, text)
                        self.results.put(("move", display_text, status_text))
                        
                        # Generate the board preview with padding so edge pieces are completely visible
                        left, top, right, bottom = self.calibration.rect
                        cell_w = (right - left) / 8.0
                        cell_h = (bottom - top) / 9.0
                        pad = int(max(32, max(cell_w, cell_h) * 0.65))

                        crop_left = max(0, left - pad)
                        crop_top = max(0, top - pad)
                        crop_right = min(frame.shape[1], right + pad)
                        crop_bottom = min(frame.shape[0], bottom + pad)

                        board_crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()
                        coords = PikafishEngine.uci_to_coords(move)
                        if coords:
                            sc, sr, ec, er = coords
                            points = grid_points(self.calibration.rect, self.calibration.slant)
                            x1 = points[sr][sc][0] - crop_left
                            y1 = points[sr][sc][1] - crop_top
                            x2 = points[er][ec][0] - crop_left
                            y2 = points[er][ec][1] - crop_top
                            cv2.arrowedLine(board_crop, (x1, y1), (x2, y2), (0, 0, 255), 4, tipLength=0.15)
                        self.results.put(("board_preview", board_crop))
                    except Exception as error:
                        # ponytail: If an illegal move or analysis error occurs, log it and retry in the next iteration.
                        self.results.put(("status", f"行棋或分析异常: {error}，正在重新计算"))
                        self.results.put(("clear_move", f"行棋或分析异常: {error}，正在重新计算"))
                        self.stop_event.wait(1.0)
                elif result.error:
                    self.results.put(("status", result.error + diagnostic_error))
                else:
                    status = "等待你落子" if state.side_to_move == player else "等待对手落子"
                    self.results.put(("status", status))
                self.stop_event.wait(POLL_INTERVAL_SECONDS)
        except Exception as error:
            self.results.put(("error", str(error)))
        finally:
            with self.engine_lock:
                engine, self.engine = self.engine, None
            if engine:
                engine.close()
            self.results.put(("stopped",))

    def best_move(self, position, movetime=5000):
        last_error = None
        for _ in range(2):
            engine = None
            try:
                with self.engine_lock:
                    if self.stop_event.is_set() or self.closing:
                        raise RuntimeError("辅助已停止")
                    if self.engine is None or not getattr(self.engine, "is_alive", lambda: True)():
                        self.engine = PikafishEngine()
                    engine = self.engine
                return engine.get_best_move(position, movetime=movetime)
            except ValueError:
                raise
            except Exception as error:
                last_error = error
                if engine:
                    with self.engine_lock:
                        if self.engine is engine:
                            self.engine = None
                    engine.close()
                if self.stop_event.is_set() or self.closing:
                    raise RuntimeError("辅助已停止") from error
        raise RuntimeError(f"Pikafish启动或分析失败: {last_error}")

    def best_advice(self, board, position, side, movetime=5000):
        last_move = None
        for _ in range(2):
            move, score = self.best_move(position, movetime)
            last_move = move
            coords = PikafishEngine.uci_to_coords(move)
            if coords:
                sc, sr, ec, er = coords
                if is_legal_xiangqi_move(board, sr, sc, er, ec, side):
                    after = [list(row) for row in board]
                    after[er][ec] = after[sr][sc]
                    after[sr][sc] = ""
                    detected = detect_move(board, after)
                    if detected and detected.side == side:
                        return move, score, PikafishEngine.get_chinese_move(
                            move, board
                        )
            with self.engine_lock:
                engine, self.engine = self.engine, None
            if engine:
                engine.close()
        raise RuntimeError(f"Pikafish返回非法着法: {last_move or '无'}")

    def poll_results(self):
        try:
            while True:
                event = self.results.get_nowait()
                if event[0] == "move":
                    self.move_var.set(event[1])
                    self.status_var.set(event[2])
                elif event[0] == "board_preview":
                    try:
                        import cv2
                        from PIL import Image, ImageTk
                        board_crop = event[1]
                        h, w = board_crop.shape[:2]
                        target_w = 240
                        target_h = int(target_w * h / w)
                        resized = cv2.resize(board_crop, (target_w, target_h))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        photo = ImageTk.PhotoImage(pil_img)
                        self.preview_photo = photo
                        self.preview_label.configure(image=photo)
                    except Exception as err:
                        self.status_var.set(f"渲染走法预览失败: {err}")
                elif event[0] == "status":
                    self.status_var.set(event[1])
                elif event[0] == "clear_move":
                    self.move_var.set("--")
                    self.status_var.set(event[1])
                    self.preview_label.configure(image="")
                    self.preview_photo = None
                elif event[0] == "turn":
                    self.side_to_move_var.set(
                        "红方" if event[1] == "w" else "黑方"
                    )
                elif event[0] == "needs_sync":
                    self.pending_recovery_board = event[1]
                    self.move_var.set("--")
                    self.preview_label.configure(image="")
                    self.preview_photo = None
                    self.side_to_move_box.configure(state="readonly")
                    self.status_var.set(
                        "请选择“局面轮到”，然后点击“同步当前棋盘”"
                    )
                elif event[0] == "error":
                    self.status_var.set(event[1])
                elif event[0] == "stopped":
                    self.worker = None
                    self.sync_button.configure(state="normal")
                    self.start_button.configure(text="停止分析", state="disabled")
                    self.player_box.configure(state="readonly")
                    self.side_to_move_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_results)

    def close(self):
        self.closing = True
        self.stop_event.set()
        with self.engine_lock:
            engine, self.engine = self.engine, None
        if engine:
            engine.close()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=12)
        if self.worker and self.worker.is_alive():
            if hasattr(self, "status_var"):
                self.status_var.set("正在安全关闭...")
            self.root.after(200, self.close)
            return
        self.root.destroy()


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    AssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
