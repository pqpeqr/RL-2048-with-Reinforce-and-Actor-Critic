import random
import secrets
import copy
from typing import Optional



Action = int    #0:up, 1: right, 2: down, 3: left


class Game2048:
    def __init__(self, size: int = 4, lg2: bool = False):
        self.size = size
        self.lg2 = lg2
        self.board: list[list[int]] = [[0] * size for _ in range(self.size)]
        self.score: int = 0
        self.rng = random.Random()
        self.new_merged = []        # for counting gain in each move
        # self.number_record = 0      # for score count

    def set_seed(self, seed: Optional[int] = None):
        # TODO: log seed
        if seed is None:
            seed = secrets.randbits(64)
        self.rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> list[list[int]]:
        self.set_seed(seed)
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.spawn()
        self.spawn()
        # self.number_record = 0
        return self.state()
        
    def spawn(self):
        empties = [(r, c) for r in range(self.size) for c in range(self.size)
                   if self.board[r][c] == 0]
        if not empties:
            # board is full, empties is []
            return
        r, c = self.rng.choice(empties)
        #90% -> 2, 10% -> 4
        self.board[r][c] = 2 if self.rng.random() < 0.9 else 4
        
    def state(self, lg2: Optional[bool] = None) -> list[list[int]]:
        if lg2 is None:
            lg2 = self.lg2
        if lg2:
            state = [[0] * self.size for _ in range(self.size)]
            for r in range(self.size):
                for c in range(self.size):
                    v = self.board[r][c]
                    state[r][c] = 0 if v == 0 else v.bit_length() - 1
            return state
        else:
            return copy.deepcopy(self.board)
    
    def row_move_left(self, row: list[int]) -> list[int]:
        cells = [x for x in row if x != 0]
        new_row: list[int] = []
        i = 0
        while i < len(cells):
            if i + 1 < len(cells) and cells[i] == cells[i + 1]:
                val = cells[i] << 1
                new_row.append(val)
                self.new_merged.append(val)
                i += 2
            else:
                new_row.append(cells[i])
                i += 1
        return new_row + [0] * (len(row) - len(new_row))   # pad 0

        
    def board_move_left(self) -> bool:
        """
        return board changed or not
        """
        is_changed = False
        new_board = []
        for row in self.board:
            new_row = self.row_move_left(row)
            new_board.append(new_row)
            if new_row != row:
                is_changed = True
        self.board = new_board
        return is_changed
            
    def rotate_clockwise(self, times: int):
        times %= 4
        for _ in range(times):
            self.board = [list(row) for row in zip(*self.board[::-1])]

    def move(self, action:Action) -> bool:
        rotations = 3 - action        #{0:3, 1:2, 2:1, 3:0}
        self.rotate_clockwise(rotations)
        is_changed = self.board_move_left()
        self.rotate_clockwise((4 - rotations) % 4)
        return is_changed
    
    def step(self, action:Action) -> tuple[list[list[int]], int, bool]:
        """
        return state, reward, is_done
        """
        # TODO: log action
        if action not in (0, 1, 2, 3):
            raise ValueError("invalid action")
        self.new_merged = []
        is_changed = self.move(action)
        reward = self.count_reward(is_changed)
        self.score += reward
        if is_changed:
            self.spawn()
        return self.state(), reward, self.is_done()
            
    def count_reward(self, is_changed: bool) -> int:
        # score for each merged cell
        reward = sum(self.new_merged)
        # # score for new biggest num
        # if self.new_merged:
        #     current_max = max(self.new_merged)
        #     if current_max > self.number_record:
        #         reward += current_max
        #         self.number_record = current_max
        # score mines for redundant step
        if not is_changed:
            reward -= 1
        return reward
    
    def is_done(self) -> bool:
        if any(0 in row for row in self.board):
            return False
        for r in range(self.size):
            for c in range(self.size):
                v = self.board[r][c]
                if  (r + 1 < self.size and self.board[r+1][c] == v) or \
                    (c + 1 < self.size and self.board[r][c+1] == v):
                    return False
        return True

    def render(self, lg2: Optional[bool] = None):
        if lg2 is None:
            lg2 = self.lg2
        state = self.state(lg2)
        width = max(4, max((len(str(x)) for row in state for x in row), default=1))
        sep = "+" + "+".join(["-" * (width)] * self.size) + "+"
        lines = [sep]
        for row in state:
            lines.append("|" + "|".join(f"{x}".rjust(width) if x else " ".rjust(width) for x in row) + "|")
            lines.append(sep)
        lines.append(f"Score: {self.score}")
        print("\n".join(lines))
    
    
# TODO: log seed, action
# TODO: max reward + Markovization + switch
# TODO: action mask + switch