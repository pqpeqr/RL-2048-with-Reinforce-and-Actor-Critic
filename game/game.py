import random
import secrets
import copy
from typing import Optional

from logger import log_append


Action = int    # 0:up, 1: right, 2: down, 3: left


class Game2048:
    def __init__(self, 
                 size: int = 4,
                 *, 
                 log_file: str = "game2048.log"):
        self.size = size
        self.log_file = log_file
        
        self.board: list[list[int]] = [[0] * size for _ in range(self.size)]
        self.score: int = 0
        self._rng = random.Random()
        
        self._new_merged: list[int] = []    # for counting gain in each move
        

    # ------pub---------
    def reset(self, seed: Optional[int] = None) -> list[list[int]]:
        self._log(f"------game reset------")
        self._set_seed(seed)
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self._spawn()
        self._spawn()
        return self.state
    
    @property
    def state(self) -> list[list[int]]:
        return copy.deepcopy(self.board)
    
    def step(self, action:Action) -> tuple[bool, list[list[int]], list[int], bool]:
        """
        return state, new_merged, is_done
        """
        if action not in (0, 1, 2, 3):
            raise ValueError("invalid action")
        self._new_merged = []
        is_changed = self._move(action)

        step_score = self._count_step_score(is_changed)
        self.score += step_score
        
        if is_changed:
            self._spawn()
        self._log(
            f"action: {action}, "
            f"changed={is_changed}, "
            f"step_score={step_score}, "
            f"score={self.score}"
        )
        return is_changed, self.state, self._new_merged, self._is_done()

    def render(self):
        state = self.state
        width = max(4, 
                    max((len(str(x)) for row in state for x in row), default=1))
        sep = "+" + "+".join(["-" * (width)] * self.size) + "+"
        lines = [sep]
        for row in state:
            lines.append(
                "|" + 
                "|".join(f"{x}".rjust(width) if x else " ".rjust(width) 
                         for x in row) + 
                "|"
            )
            lines.append(sep)
        lines.append(f"Score: {self.score}")
        print("\n".join(lines))

    def get_action_mask(self) -> list[int]:
        mask = []
        for action in (0, 1, 2, 3):
            mask.append(1 if self._can_change_with_action(action) else 0)
        return mask

    # ------priv---------
    def _log(self, msg: str) -> None:
        try:
            log_append(msg, filepath=self.log_file)
        except Exception:
            pass
    
    def _set_seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = secrets.randbits(64)
        self._rng.seed(seed)
        self._log(f"seed set: {seed}")
    
    def _spawn(self):
        empties = [(r, c) for r in range(self.size) for c in range(self.size)
                   if self.board[r][c] == 0]
        if not empties:
            # board is full, empties is []
            return
        r, c = self._rng.choice(empties)
        #90% -> 2, 10% -> 4
        self.board[r][c] = 2 if self._rng.random() < 0.9 else 4
    
    def _row_move_left(self, row: list[int]) -> list[int]:
        cells = [x for x in row if x != 0]
        new_row: list[int] = []
        i = 0
        while i < len(cells):
            if i + 1 < len(cells) and cells[i] == cells[i + 1]:
                val = cells[i] << 1
                new_row.append(val)
                self._new_merged.append(val)
                i += 2
            else:
                new_row.append(cells[i])
                i += 1
        return new_row + [0] * (len(row) - len(new_row))   # pad 0

    def _board_move_left(self) -> bool:
        """
        return board changed or not
        """
        is_changed = False
        new_board = []
        for row in self.board:
            new_row = self._row_move_left(row)
            new_board.append(new_row)
            if new_row != row:
                is_changed = True
        self.board = new_board
        return is_changed
            
    def rotate_clockwise(self, times: int):
        times %= 4
        for _ in range(times):
            self.board = [list(row) for row in zip(*self.board[::-1])]

    def _move(self, action:Action) -> bool:
        rotations = 3 - action        #{0:3, 1:2, 2:1, 3:0}
        self.rotate_clockwise(rotations)
        is_changed = self._board_move_left()
        self.rotate_clockwise((4 - rotations) % 4)
        return is_changed
    
    def _count_step_score(self) -> int:
        # score for each merged cell
        step_score = sum(self._new_merged)
        return step_score
    
    def _is_done(self) -> bool:
        if any(0 in row for row in self.board):
            return False
        for r in range(self.size):
            for c in range(self.size):
                v = self.board[r][c]
                if  (r + 1 < self.size and self.board[r+1][c] == v) or \
                    (c + 1 < self.size and self.board[r][c+1] == v):
                    return False
        self._log("------game done------")
        return True
    
    @staticmethod
    def _row_move_left_preview(row: list[int]) -> list[int]:
        cells = [x for x in row if x != 0]
        new_row: list[int] = []
        i = 0
        while i < len(cells):
            if i + 1 < len(cells) and cells[i] == cells[i + 1]:
                val = cells[i] << 1
                new_row.append(val)
                i += 2
            else:
                new_row.append(cells[i])
                i += 1
        return new_row + [0] * (len(row) - len(new_row))
    
    @staticmethod
    def _board_move_left_preview(board: list[list[int]]) -> tuple[bool, list[list[int]]]:
        is_changed = False
        new_board: list[list[int]] = []
        for row in board:
            new_row = Game2048._row_move_left_preview(row)
            new_board.append(new_row)
            if new_row != row:
                is_changed = True
        return is_changed, new_board
    
    @staticmethod
    def _rotate_clockwise_preview(board: list[list[int]], times: int) -> list[list[int]]:
        times %= 4
        b = [row[:] for row in board]       # "deep copy"
        for _ in range(times):
            b = [list(row) for row in zip(*b[::-1])]
        return b
        
    def _can_change_with_action(self, action: Action) -> bool:
        rotations = 3 - action
        temp = self._rotate_clockwise_preview(self.board, rotations)
        changed, _ = self._board_move_left_preview(temp)
        return changed
