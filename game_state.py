from dataclasses import dataclass

from board_recognition import board_to_fen, detect_move, move_to_uci, normalize_board


@dataclass(frozen=True)
class Snapshot:
    board: tuple
    side_to_move: str
    moves_length: int


class GameState:
    @classmethod
    def start(cls, board, side_to_move, player, base_fen=None):
        return cls(board, side_to_move, player, base_fen)

    def __init__(self, board, side_to_move, player, base_fen=None):
        if side_to_move not in {"w", "b"} or player not in {"w", "b"}:
            raise ValueError("行棋方必须是w或b")
        self.board = normalize_board(board)
        self.side_to_move = side_to_move
        self.player = player
        self.base_fen = base_fen or board_to_fen(self.board, side_to_move)
        self.moves = []
        self.snapshots = [Snapshot(self.board, side_to_move, 0)]
        self.position_version = 0
        self.last_analyzed_version = None

    def apply_board(self, board):
        board = normalize_board(board)
        move = detect_move(self.board, board)
        if move is None or move.side != self.side_to_move:
            return False
        self.moves.append(move_to_uci(move))
        self.board = board
        self.side_to_move = "b" if self.side_to_move == "w" else "w"
        self.snapshots.append(Snapshot(board, self.side_to_move, len(self.moves)))
        self.position_version += 1
        return True

    def restore_snapshot(self, board):
        board = normalize_board(board)
        match = next(
            (
                (index, snapshot)
                for index, snapshot in reversed(list(enumerate(self.snapshots[:-1])))
                if snapshot.board == board
            ),
            None,
        )
        if match is None:
            return False
        index, snapshot = match
        self.board = snapshot.board
        self.side_to_move = snapshot.side_to_move
        self.moves = self.moves[: snapshot.moves_length]
        self.snapshots = self.snapshots[: index + 1]
        self.position_version += 1
        return True

    def resync(self, board, side_to_move, base_fen=None):
        if side_to_move not in {"w", "b"}:
            raise ValueError("行棋方必须是w或b")
        self.board = normalize_board(board)
        self.side_to_move = side_to_move
        self.base_fen = base_fen or board_to_fen(self.board, side_to_move)
        self.moves = []
        self.snapshots = [Snapshot(self.board, side_to_move, 0)]
        self.position_version += 1

    def needs_analysis(self):
        return (
            self.side_to_move == self.player
            and self.last_analyzed_version != self.position_version
        )

    def mark_analyzed(self):
        self.last_analyzed_version = self.position_version

    def position_command(self):
        suffix = f" moves {' '.join(self.moves)}" if self.moves else ""
        return f"position fen {self.base_fen}{suffix}"
