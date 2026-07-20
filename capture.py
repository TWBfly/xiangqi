import ctypes
from ctypes import wintypes
import os
import platform
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


def parse_adb_devices(output):
    return [
        parts[0]
        for line in output.splitlines()
        if len(parts := line.split()) == 2 and parts[1] == "device"
    ]


def enable_dpi_awareness():
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def find_adb(saved_path=None):
    candidates = [saved_path, os.environ.get("XIANGQI_ADB")]
    program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    relative_paths = (
        "LDPlayer/LDPlayer9/adb.exe",
        "LDPlayer/LDPlayer4.0/adb.exe",
        "TxGameAssistant/AppMarket/ADB/adb.exe",
        "TxGameAssistant/ui/adb.exe",
    )
    candidates.extend(
        str(Path(root) / relative)
        for root in program_files
        if root
        for relative in relative_paths
    )
    candidates.extend(
        f"{drive}:/{relative}"
        for drive in ("C", "D", "E")
        for relative in relative_paths
    )
    candidates.append(shutil.which("adb"))
    return next((str(Path(path)) for path in candidates if path and Path(path).is_file()), None)


def adb_devices(adb, timeout=3.0):
    result = subprocess.run(
        [adb, "devices"], check=True, capture_output=True, text=True, timeout=timeout
    )
    return parse_adb_devices(result.stdout)


def adb_frame(adb, serial, timeout=3.0):
    data = subprocess.run(
        [adb, "-s", serial, "exec-out", "screencap", "-p"],
        check=True,
        capture_output=True,
        timeout=timeout,
    ).stdout
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("模拟器截图不是有效PNG")
    return validate_frame(frame)


def validate_frame(frame, expected_shape=None):
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 32:
        raise RuntimeError("模拟器截图无效")
    if expected_shape and tuple(frame.shape[:2]) != tuple(expected_shape):
        raise RuntimeError("模拟器截图尺寸已变化")
    if float(frame.std()) < 1.0:
        raise RuntimeError("模拟器截图为空白")
    return frame


def select_emulator_windows(windows):
    game_names = ("jj象棋",)
    tencent_names = ("腾讯手游助手", "gameloop")
    emulator_names = ("雷电", "ldplayer") + tencent_names
    matched = [
        window
        for window in windows
        if any(name in window[1].lower() for name in game_names + emulator_names)
    ]
    if any(any(name in title.lower() for name in game_names) for _, title in matched):
        matched = [
            window
            for window in matched
            if not any(name in window[1].lower() for name in tencent_names)
        ]
    return sorted(
        matched,
        key=lambda window: 0
        if any(name in window[1].lower() for name in game_names)
        else 1,
    )


def list_emulator_windows():
    if platform.system() != "Windows":
        return []

    user32 = ctypes.windll.user32
    titles = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowTextW.restype = ctypes.c_int

    @callback_type
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        titles.append((int(hwnd), title))
        return True

    user32.EnumWindows(callback, 0)
    return select_emulator_windows(titles)


def window_frame(hwnd):
    if platform.system() != "Windows":
        raise RuntimeError("窗口截图只支持Windows")

    user32 = ctypes.windll.user32
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetWindowRect.restype = wintypes.BOOL
    if user32.IsIconic(hwnd):
        raise RuntimeError("所选窗口已最小化，请刷新设备后选择JJ象棋")
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("无法读取模拟器窗口位置")
    if rect.right <= rect.left or rect.bottom <= rect.top:
        raise RuntimeError("模拟器窗口已最小化")

    # ponytail: Get system metrics of virtual screen to clamp window boundaries safely,
    # preventing Pillow ImageGrab GDI failures (Errno 22) on out-of-bounds or negative coords.
    v_left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
    v_top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
    v_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
    v_height = user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN

    if v_width == 0 or v_height == 0:
        v_left, v_top = 0, 0
        v_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        v_height = user32.GetSystemMetrics(1) # SM_CYSCREEN

    v_right = v_left + v_width
    v_bottom = v_top + v_height

    left = max(v_left, min(rect.left, v_right))
    top = max(v_top, min(rect.top, v_bottom))
    right = max(v_left, min(rect.right, v_right))
    bottom = max(v_top, min(rect.bottom, v_bottom))

    if right - left <= 0 or bottom - top <= 0:
        raise RuntimeError("模拟器窗口完全在可见屏幕区域之外")

    image = None
    # 1. 尝试直接以限制后的 bbox 和多屏模式抓取
    try:
        image = np.asarray(ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True))
    except Exception:
        pass

    # 2. 如果依然失败，使用 100% 稳定的整屏抓取，并在内存中进行切片以彻底避免 GDI 参数报错
    if image is None:
        try:
            full_image = ImageGrab.grab(all_screens=True)
            offset_x, offset_y = v_left, v_top
        except Exception:
            full_image = ImageGrab.grab(all_screens=False)
            offset_x, offset_y = 0, 0

        full_width, full_height = full_image.size
        local_left = max(0, min(left - offset_x, full_width))
        local_top = max(0, min(top - offset_y, full_height))
        local_right = max(0, min(right - offset_x, full_width))
        local_bottom = max(0, min(bottom - offset_y, full_height))

        if local_right - local_left <= 0 or local_bottom - local_top <= 0:
            raise RuntimeError("模拟器窗口截图区域无效")

        cropped = full_image.crop((local_left, local_top, local_right, local_bottom))
        image = np.asarray(cropped)

    return validate_frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
