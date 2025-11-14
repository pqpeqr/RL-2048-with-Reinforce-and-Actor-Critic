import logging

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


ObsMode = Literal["raw", "log2"]
RewardMode = Literal["sum", "log2", "bonus"]


from game.game2048 import Game2048, Action


@dataclass
class Game2048EnvConfig:
    size: int = 4
    obs_mode: ObsMode = "log2"             # raw / log2
    reward_mode: RewardMode = "sum"        # sum / log2 / bonus
    bonus_first_merge: bool = True         # T / F
    bonus_value: float = 1.0               # bonus ratio
    use_action_mask: bool = True           # T / F
    invalid_action_penalty: float = -1.0   # penalty when not using action musk
    max_steps: int = None                  # max step in an episode
    log_level: int = logging.INFO
    


class Game2048Env(gym.Env):

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Game2048EnvConfig | None = None):
        super().__init__()

        self.config = config or Game2048EnvConfig()
        self.game = Game2048(size=self.config.size)

        self._step_count = 0

        # max tile record for bonus reward
        self._max_tile_seen: int = 4

        # logger
        self._logger = logging.getLogger(__name__ + ".Game2048Env")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
        self._logger.setLevel(self.config.log_level)
        
        logging.getLogger("game2048.Game2048").disabled = True      # mute Game's logger

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = self._build_observation_space()

    
    def _build_observation_space(self) -> spaces.Space:
        size = self.config.size

        # the max theoretical number for 4x4 2048 is 2^17, 2^16 is fine for us
        if self.config.obs_mode == "raw":
            # 0, 2, 4, 8, ...
            board_space = spaces.Box(
                low=0,
                high=2**16,
                shape=(size, size),
                dtype=np.float32,
            )
        else:
            # log2: 0, 1, 2, 3, ...
            board_space = spaces.Box(
                low=0.0,
                high=16.0,
                shape=(size, size),
                dtype=np.float32,
            )

        if self.config.use_action_mask:
            # Dict：{"board": board, "action_mask": mask}
            obs_space = spaces.Dict(
                {
                    "board": board_space,
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.action_space.n,),
                        dtype=np.int8,
                    ),
                }
            )
        else:
            obs_space = board_space

        return obs_space
    
    
    def _preprocess_board(self, board: np.ndarray) -> np.ndarray:
        board = board.astype(np.float32, copy=True)

        if self.config.obs_mode == "raw":
            return board

        # log2 mode; 0 -> 0
        non_zero = board > 0
        board[non_zero] = np.log2(board[non_zero])
        return board
    
    
    def _get_action_mask(self) -> np.ndarray:
        mask_list = self.game.get_action_mask()         # list[int] of length 4
        return np.array(mask_list, dtype=np.int8)
    
    
    def _get_obs(self):
        board = self.game.board
        
        processed = self._preprocess_board(board)

        if self.config.use_action_mask:
            mask = self._get_action_mask()
            return {
                "board": processed,
                "action_mask": mask,
            }
        else:
            return processed
    
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        self._max_tile_seen = 4
        
        state = self.game.reset(seed=seed)

        obs = self._get_obs()

        info = {
            "score": self.game.score,
            "raw_state": state,
        }

        self._logger.info("Env reset with seed=%s, initial_score=%s", seed, self.game.score)

        return obs, info
    
    
    def _compute_reward(
        self,
        merged: list[int],
        done: bool,
        invalid_action: bool,
    ) -> float:
        cfg = self.config

        # when not using action mask, give penalty for unchanged action
        if not cfg.use_action_mask and invalid_action:
            return cfg.invalid_action_penalty

        # base reward
        base = sum(merged)

        if cfg.reward_mode == "sum":
            reward = float(base)

        elif cfg.reward_mode == "log2":
            # log 2 reward
            reward = 0.0
            for v in merged:
                if v > 0:
                    reward += float(np.log2(v))

        elif cfg.reward_mode == "bonus":
            # base reward no change
            reward = float(base)

            if cfg.bonus_first_merge and merged:
                max_merged = max(merged)

                # only when >=8, to avoid disturbance form 4 at beginning of game
                if max_merged >= 8 and max_merged > self._max_tile_seen:
                    reward += cfg.bonus_value * float(np.log2(max_merged))
                    self._max_tile_seen = max_merged
        else:
            # should not reach here
            reward = float(base)

        return reward
    
    
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1

        # one step forward
        is_changed, state, merged, is_done = self.game.step(action)

        # is it changed?
        invalid_action = (not is_changed) and (not is_done)

        # count reward
        reward = self._compute_reward(merged, is_done, invalid_action)

        # terminated / truncated
        terminated = bool(is_done)
        if self.config.max_steps is None:
            truncated = False
        else:
            truncated = (
                self._step_count >= self.config.max_steps
                and not terminated
            )
        obs = self._get_obs()

        info = {
            "score": self.game.score,
            "raw_state": state,
            "merged": merged,
            "invalid_action": invalid_action,
            "step_index": self._step_count,
        }

        # logging
        self._logger.info(
            "step=%d, action=%d, changed=%s, merged=%s, reward=%.3f, score=%d, done=%s, truncated=%s",
            self._step_count,
            action,
            is_changed,
            merged,
            reward,
            self.game.score,
            terminated,
            truncated,
        )

        return obs, reward, terminated, truncated, info


    def render(self, mode: str = "human"):
        text = self.game.render()
        if mode == "human":
            print(text)
            return None
        elif mode == "ansi":
            return text
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")
    
    