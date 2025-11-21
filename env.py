import logging

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


ObsMode = Literal["raw", "log2"]
RewardMode = Literal["sum", "log2"]
BonusMode = Literal["off", "raw", "log2"]


from game.game2048 import Game2048, Action


@dataclass
class Game2048EnvConfig:
    size: int = 4
    obs_mode: ObsMode = "raw"               # raw / log2 / onehot
    obs_log2_scale: float = 1.0             # scale factor for log2 observation
    reward_mode: RewardMode = "sum"         # sum / log2
    reward_scale: float = 1.0               # scale factor for reward
    bonus_mode: BonusMode = "off"           # off / raw / log2
    bonus_scale: float = 1.0                # scale factor for bonus
    step_reward: float = 0.0                # reward for each step
    endgame_penalty: float = 0.0            # penalty at the end of game
    use_action_mask: bool = True            # T / F
    invalid_action_penalty: float = -1.0    # penalty when not using action musk
    max_steps: int = 1024                   # max step in an episode
    


class Game2048Env(gym.Env):

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Game2048EnvConfig | None = None) -> None:
        super().__init__()

        self.config = config or Game2048EnvConfig()
        self.game = Game2048(size=self.config.size)
        
        # max tile number for observation space(2^n for raw / n for log2 / [n + 1] for onehot)
        # if > 24 and using raw, may need to use float64 to avoid overflow
        self._max_num: float = 16.0

        self._step_count: int = 0

        # max tile record for bonus reward
        self.max_tile_seen: int = 4

        # logger
        self._logger = logging.getLogger(__name__ + ".Game2048Env")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
        

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = self._build_observation_space()
    
    
    @property
    def state(self) -> list[list[int]]:
        return self.game.state

    
    def _build_observation_space(self) -> spaces.Space:
        size = self.config.size

        # the max theoretical number for 4x4 2048 is 2^17, 2^16 is fine for us
        if self.config.obs_mode == "raw":
            # 0, 2, 4, 8, ...
            board_space = spaces.Box(
                low=0,
                high=2**self._max_num,
                shape=(size, size),
                dtype=np.float32,
            )
        elif self.config.obs_mode == "log2" or self.config.obs_mode == "onehot":
            # log2: 0, 1, 2, 3, ...
            board_space = spaces.Box(
                low=0.0,
                high=self._max_num,
                shape=(size, size),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported obs_mode: {self.config.obs_mode}")

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
        elif self.config.obs_mode == "log2":
            # log2 mode; 0 -> 0
            non_zero = board > 0
            board[non_zero] = np.log2(board[non_zero])
            board *= self.config.obs_log2_scale
            return board
        elif self.config.obs_mode == "onehot":
            non_zero = board > 0
            exponents = np.zeros_like(board, dtype=np.int32)
            exponents[non_zero] = np.log2(board[non_zero]).astype(np.int32)
            max_exp = int(self._max_num)
            num_channels = max_exp + 1
            # trick: use np.eye to get onehot encoding
            one_hot = np.eye(num_channels, dtype=np.float32)[exponents]
            return one_hot


    
    def _get_action_mask(self) -> np.ndarray:
        mask_list = self.game.get_action_mask()         # list[int] of length 4
        return np.array(mask_list, dtype=np.int8)
    
    
    def _get_obs(self) -> np.ndarray | dict[str, Any]:
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
        self.max_tile_seen = 4
        
        state = self.game.reset(seed=seed)

        obs = self._get_obs()

        info = {
            "score": self.game.score,
            "raw_state": state,
        }
        
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
                    
        else:
            raise ValueError(f"Unsupported reward mode: {cfg.reward_mode}")
        
        reward *= cfg.reward_scale
        
        # bonus reward
        bonus = 0.0
        max_merged = max(merged, default=0)
        # only when >=8, to avoid disturbance form 4 at beginning of game
        if max_merged >= 8 and max_merged > self.max_tile_seen:
            if cfg.bonus_mode == "off":
                pass
            elif cfg.bonus_mode == "raw":
                bonus = float(max_merged)
            elif cfg.bonus_mode == "log2":
                bonus = float(np.log2(max_merged))
            else:
                raise ValueError(f"Unsupported bonus mode: {cfg.bonus_mode}")
            self.max_tile_seen = max_merged
            bonus *= cfg.bonus_scale
            reward += bonus
        
        # step reward
        reward += cfg.step_reward
        
        # endgame penalty
        if done and cfg.endgame_penalty != 0.0:
            reward += cfg.endgame_penalty
        
        return reward
    
    
    def step(self, action: int) -> tuple[np.ndarray | dict[str, Any], float, bool, bool, dict[str, Any]]:
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
        self._logger.debug(
            f"reward={reward:.3f}, truncated={truncated}"
        )

        return obs, reward, terminated, truncated, info


    def render(self, mode: str = "human") -> str | None:
        text = self.game.render()
        if mode == "human":
            print(text)
            return None
        elif mode == "ansi":
            return text
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")
    
    