import logging

from dataclasses import dataclass
from typing import Literal, Optional

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

    def __init__(self, config: Optional[Game2048EnvConfig] = None):
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
        pass
    
    
    def reset(self):
        pass
    
    
    def step(self):
        pass


    def render(self, mode: str = "human"):
        text = self.game.render()
        if mode == "human":
            print(text)
            return None
        elif mode == "ansi":
            return text
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")
    
    