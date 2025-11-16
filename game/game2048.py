import secrets
import logging



import numpy as np


Action = int    # 0:up, 1: right, 2: down, 3: left

class Game2048:
    def __init__(self, 
                 size: int = 4):
        self.size = size
        
        self.board: np.ndarray = np.zeros((size, size), dtype=np.int64)
        self.step_count: int = 0
        self.score: int = 0
        self._rng: np.random.Generator = np.random.default_rng()
        
        self._new_merged: list[int] = []    # for counting gain in each move
        
        # logging
        self._logger = logging.getLogger(__name__+".Game2048")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
    
    def reset(self, seed: int | None = None) -> list[list[int]]:
        self._log(f"------game reset------")
        self._set_seed(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.step_count = 0
        self.score = 0
        self._spawn()
        self._spawn()
        return self.state
    
    @property
    def state(self) -> list[list[int]]:
        return self.board.tolist()
    
    def step(self, action: Action) -> tuple[bool, list[list[int]], list[int], bool]:
        """
        return is_changed, state, new_merged, is_done
        """
        if action not in (0, 1, 2, 3):
            raise ValueError("invalid action")

        self.step_count += 1
        
        self._new_merged = []
        is_changed = self._move(action)
        is_done = self._is_done()

        step_score = self._count_step_score()
        self.score += step_score

        if is_changed:
            self._spawn()

        self._log(
            f"action: {action}, "
            f"changed={is_changed}, "
            f"step_score={step_score}, "
            f"score={self.score}"
        )
        return is_changed, self.state, self._new_merged.copy(), is_done
    
    
    def render(self) -> str:
        state = self.state
        width = max(
            4,
            max((len(str(x)) for row in state for x in row), default=1)
        )
        sep = "+" + "+".join(["-" * width] * self.size) + "+"
        lines = [sep]
        for row in state:
            lines.append(
                "|" +
                "|".join(
                    f"{x}".rjust(width) if x else " ".rjust(width)
                    for x in row
                ) +
                "|"
            )
            lines.append(sep)
        # lines.append(f"Score: {self.score}")
        # print("\n".join(lines))
        return "\n".join(lines)
        
    def get_action_mask(self) -> list[int]:
        mask: list[int] = []
        for action in (0, 1, 2, 3):
            mask.append(1 if self._can_change_with_action(action) else 0)
        return mask
    
    def _log(self, msg: str) -> None:
        try:
            self._logger.info(msg)
        except Exception:
            pass
        
    def _set_seed(self, seed: int | None = None):
        if seed is None:
            seed = secrets.randbits(64)
        self._rng = np.random.default_rng(seed)
        self._log(f"seed set: {seed}")

    def _spawn(self) -> None:
        empties = np.argwhere(self.board == 0)
        if empties.size == 0:
            return
        
        idx = self._rng.integers(len(empties))
        r, c = empties[idx]

        # 90% -> 2, 10% -> 4
        val = 2 if self._rng.random() < 0.9 else 4
        self.board[r, c] = val

    def _row_move_left(self, row: np.ndarray) -> np.ndarray:
        cells = row[row != 0]
        new_row = np.zeros_like(row)

        write_idx = 0
        i = 0
        n = len(cells)
        while i < n:
            if i + 1 < n and cells[i] == cells[i + 1]:
                val = int(cells[i] << 1)
                new_row[write_idx] = val
                self._new_merged.append(val)
                i += 2
            else:
                new_row[write_idx] = int(cells[i])
                i += 1
            write_idx += 1
        return new_row
    
    def _board_move_left(self) -> bool:
        is_changed = False
        new_board = np.zeros_like(self.board)

        for r in range(self.size):
            old_row = self.board[r]
            new_row = self._row_move_left(old_row)
            new_board[r] = new_row
            if not np.array_equal(new_row, old_row):
                is_changed = True

        self.board = new_board
        return is_changed

    def rotate_clockwise(self, times: int) -> None:
        times %= 4
        if times:
            self.board = np.rot90(self.board, k=-times)

    def _move(self, action: Action) -> bool:
        # {0:3, 1:2, 2:1, 3:0}
        rotations = 3 - action
        
        self.rotate_clockwise(rotations)
        is_changed = self._board_move_left()
        self.rotate_clockwise((4 - rotations) % 4)
        return is_changed
    
    def _count_step_score(self) -> int:
        # score for each merged cell
        step_score = sum(self._new_merged)
        return step_score
    
    def _is_done(self) -> bool:
        if (self.board == 0).any():
            return False

        for r in range(self.size):
            for c in range(self.size):
                v = self.board[r, c]
                if (
                    r + 1 < self.size and self.board[r + 1, c] == v
                ) or (
                    c + 1 < self.size and self.board[r, c + 1] == v
                ):
                    return False

        self._log("------game done------")
        return True
    
    @staticmethod
    def _row_move_left_preview(row: list[int]) -> list[int]:
        arr = np.array(row, dtype=int)
        cells = arr[arr != 0]
        new_arr = np.zeros_like(arr)

        write_idx = 0
        i = 0
        n = len(cells)
        while i < n:
            if i + 1 < n and cells[i] == cells[i + 1]:
                val = int(cells[i] << 1)
                new_arr[write_idx] = val
                i += 2
            else:
                new_arr[write_idx] = int(cells[i])
                i += 1
            write_idx += 1

        return new_arr.tolist()
    
    @staticmethod
    def _board_move_left_preview(board: list[list[int]]) -> tuple[bool, list[list[int]]]:
        is_changed = False
        b = np.array(board, dtype=int)
        new_b = np.zeros_like(b)

        for r in range(b.shape[0]):
            old_row = b[r]
            new_row = np.array(Game2048._row_move_left_preview(old_row.tolist()), dtype=int)
            new_b[r] = new_row
            if not np.array_equal(new_row, old_row):
                is_changed = True

        return is_changed, new_b.tolist()
    
    @staticmethod
    def _rotate_clockwise_preview(board: list[list[int]], times: int) -> list[list[int]]:
        times %= 4
        b = np.array(board, dtype=int)
        if times:
            b = np.rot90(b, k=-times)
        return b.tolist()
    
    def _can_change_with_action(self, action: Action) -> bool:
        rotations = 3 - action
        temp_board = self._rotate_clockwise_preview(self.state, rotations)
        changed, _ = self._board_move_left_preview(temp_board)
        return changed
