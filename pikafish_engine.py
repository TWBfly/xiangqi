import os
import platform
import queue
import subprocess
import sys
import threading
import time


import shutil

from board_recognition import fen_to_board


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)


def get_safe_engine_paths(binary_path, nnue_path):
    """
    检查路径中是否包含非 ASCII 字符。
    如果是 Windows 系统且包含非 ASCII 字符，将 binary 和 nnue 复制到纯 ASCII 且有写权限的系统目录中（C:\\Users\\Public\\XiangqiAssistant），
    返回安全的可执行文件路径、影子 nnue 路径和安全工作目录。
    """
    def is_ascii(s):
        try:
            s.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    binary_abs = os.path.abspath(binary_path)
    nnue_abs = os.path.abspath(nnue_path) if nnue_path else None
    
    need_shadow = not is_ascii(binary_abs) or (nnue_abs and not is_ascii(nnue_abs))
    
    if need_shadow and platform.system() == "Windows":
        shadow_dir = "C:\\Users\\Public\\XiangqiAssistant"
        try:
            try:
                # Terminate any running pikafish.exe processes to release potential file locks
                subprocess.run(
                    ["taskkill", "/F", "/IM", "pikafish.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                time.sleep(0.1)  # Brief pause to let OS release handles
            except Exception:
                pass

            os.makedirs(shadow_dir, exist_ok=True)
            shadow_binary = os.path.join(shadow_dir, os.path.basename(binary_abs))
            
            def safe_copy(src, dst):
                if not os.path.exists(dst) or os.path.getsize(src) != os.path.getsize(dst):
                    shutil.copy2(src, dst)
            
            safe_copy(binary_abs, shadow_binary)
            
            shadow_nnue = None
            if nnue_abs and os.path.exists(nnue_abs):
                shadow_nnue = os.path.join(shadow_dir, os.path.basename(nnue_abs))
                safe_copy(nnue_abs, shadow_nnue)
                
            return shadow_binary, shadow_nnue, shadow_dir
        except Exception:
            pass
            
    return binary_abs, nnue_abs, os.path.dirname(binary_abs)


class PikafishEngine:
    def __init__(self, binary_path=None, startup_timeout=10.0):
        binary_name = "pikafish.exe" if platform.system() == "Windows" else "pikafish_bin"
        self.binary_path = binary_path or resource_path(binary_name)
        self.process = None
        self._lines = queue.Queue()
        self._lock = threading.Lock()
        self._start(startup_timeout)

    def _start(self, timeout):
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"Pikafish不存在: {self.binary_path}")
        
        nnue_path = resource_path("pikafish.nnue")
        safe_binary, safe_nnue, safe_cwd = get_safe_engine_paths(self.binary_path, nnue_path)

        self.process = subprocess.Popen(
            [safe_binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=safe_cwd,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            ),
        )
        try:
            threading.Thread(target=self._read_output, daemon=True).start()
            self.send_command("uci")
            uci_lines = self._read_until(lambda line: line == "uciok", timeout)
            self._configure_strength(uci_lines)
            
            if safe_nnue and os.path.isfile(safe_nnue):
                # ponytail: To avoid spaces or special characters in absolute paths causing
                # engine load failures, we use the safe relative path 'pikafish.nnue'.
                # The engine automatically resolves this relative to its own binary directory.
                self.send_command("setoption name EvalFile value pikafish.nnue")
            self.send_command("isready")
            self._read_until(lambda line: line == "readyok", timeout)
        except Exception:
            self.close()
            raise

    def _read_output(self):
        try:
            for line in self.process.stdout:
                self._lines.put(line.strip())
        finally:
            self._lines.put(None)

    def _read_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        lines = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("等待Pikafish响应超时")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("等待Pikafish响应超时") from error
            if line is None:
                code = self.process.poll() if self.process else None
                if code is None and self.process:
                    try:
                        code = self.process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                if isinstance(code, int):
                    code_text = f"{code} / 0x{code & 0xFFFFFFFF:08X}"
                    # ponytail: Help diagnose common CPU AVX2 instruction compatibility crashes
                    if code in (-1073741795, 0xC000001D):
                        code_text += " (非法指令，您的 CPU 很可能不支持当前 Pikafish 编译版本所需的 AVX2 指令集，请更换为通用兼容版引擎)"
                else:
                    code_text = "未知"
                detail = " | ".join(lines[-3:]) or "引擎未输出错误详情"
                raise RuntimeError(
                    f"Pikafish进程已退出（退出码 {code_text}；最后输出：{detail}）"
                )
            lines.append(line)
            if predicate(line):
                return lines

    def _configure_strength(self, uci_lines):
        option_lines = {
            line.split(" type ", 1)[0].removeprefix("option name "): line
            for line in uci_lines
            if line.startswith("option name ") and " type " in line
        }

        def maximum(name, fallback):
            words = option_lines[name].split()
            try:
                return int(words[words.index("max") + 1])
            except (ValueError, IndexError):
                return fallback

        if "Threads" in option_lines:
            threads = min(
                maximum("Threads", 1),
                8,
                max(1, os.cpu_count() or 1),
            )
            self.send_command(f"setoption name Threads value {threads}")
        if "Hash" in option_lines:
            self.send_command(
                f"setoption name Hash value {min(maximum('Hash', 512), 512)}"
            )
        if "MultiPV" in option_lines:
            self.send_command("setoption name MultiPV value 1")

    def send_command(self, command):
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Pikafish未运行")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def is_alive(self):
        return bool(self.process and self.process.poll() is None)

    def _clear_queue(self):
        # ponytail: Clear leftover lines in the queue from previous search runs
        if hasattr(self, "_lines"):
            while not self._lines.empty():
                try:
                    self._lines.get_nowait()
                except queue.Empty:
                    break

    def get_best_move(self, position, movetime=5000):
        movetime = int(movetime)
        position_command = (
            position if position.startswith("position fen ") else f"position fen {position}"
        )
        fen = position_command.removeprefix("position fen ").split(" moves ", 1)[0]
        try:
            fen_to_board(fen)
        except ValueError as error:
            raise ValueError(f"拒绝发送非法局面给Pikafish: {error}") from error
        with self._lock:
            self._clear_queue()
            self.send_command("isready")
            self._read_until(lambda line: line == "readyok", 5.0)

            self.send_command(position_command)
            self.send_command(f"go movetime {movetime}")
            lines = self._read_until(
                lambda line: line.startswith("bestmove "),
                max(5.0, movetime / 1000.0 + 5.0),
            )
        best_line = lines[-1].split()
        bestmove = best_line[1] if len(best_line) > 1 and best_line[1] != "(none)" else None
        score = "0.00"
        depth = None
        nps = None

        for line in reversed(lines):
            parts = line.split()
            if "depth" in parts and depth is None:
                try:
                    idx = parts.index("depth")
                    depth = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            if "nps" in parts and nps is None:
                try:
                    idx = parts.index("nps")
                    nps = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            if "score cp" in line and score == "0.00":
                try:
                    score = f"{int(line.split('score cp', 1)[1].split()[0]) / 100:+.2f}"
                except (ValueError, IndexError):
                    pass
            elif "score mate" in line and score == "0.00":
                mate_val = line.split("score mate", 1)[1].split()[0]
                score = f"Mate {mate_val}"

        if depth is not None:
            nps_str = f" | {nps/1e6:.1f}M NPS" if nps and nps >= 1000000 else (f" | {nps/1e3:.0f}K NPS" if nps else "")
            score = f"{score} (深度:{depth}层{nps_str})"

        return bestmove, score

    def stop_search(self):
        process = self.process
        if not process or process.poll() is not None:
            return
        try:
            process.stdin.write("stop\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def close(self):
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self):
        process, self.process = self.process, None
        if not process:
            return
        try:
            if process.poll() is None:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.wait(timeout=2)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            try:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass

    @staticmethod
    def uci_to_coords(move):
        if not move or len(move) < 4:
            return None
        try:
            values = ord(move[0]) - 97, 9 - int(move[1]), ord(move[2]) - 97, 9 - int(move[3])
        except ValueError:
            return None
        return values if all(0 <= value <= limit for value, limit in zip(values, (8, 9, 8, 9))) else None

    @staticmethod
    def get_chinese_move(move, board):
        coords = PikafishEngine.uci_to_coords(move)
        if not coords:
            return move
        sc, sr, ec, er = coords
        if isinstance(board, tuple) and len(board) == 2 and isinstance(board[0], (tuple, list)):
            board = board[0]
        piece = board[sr][sc]
        if not piece:
            return move

        red = piece.isupper()
        kind = piece.upper()
        names = {"K": "帅" if red else "将", "A": "仕" if red else "士", "B": "相" if red else "象", "N": "马", "R": "车", "C": "炮", "P": "兵" if red else "卒"}
        red_cols = "九八七六五四三二一"
        black_cols = "123456789"
        cols = red_cols if red else black_cols

        same_file = [row for row in range(10) if board[row][sc] == piece]
        if len(same_file) == 2:
            front = min(same_file) if red else max(same_file)
            prefix = ("前" if sr == front else "后") + names[kind]
        else:
            prefix = names[kind] + cols[sc]

        if sr == er:
            return prefix + "平" + cols[ec]
        forward = sr > er if red else sr < er
        direction = "进" if forward else "退"
        action = cols[ec] if kind in "NBA" else (red_cols[9 - abs(sr - er)] if red else str(abs(sr - er)))
        return prefix + direction + action
