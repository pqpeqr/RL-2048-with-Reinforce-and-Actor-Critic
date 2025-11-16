import numpy as np

from env import Game2048Env
from typing import Literal, Any
from dataclasses import dataclass

import logging

from MLP import (
    MLPConfig,
    encode_observation,
    init_model_params_0layer,
    forward_logits_0layer,
    logits_to_probs,
)



BaselineMode = Literal["off", "each", "batch"]

@dataclass
class ReinforceAgentConfig:
    gamma: float = 1                            # [0, 1]
    learning_rate: float = 1e-3                 # 
    baseline_mode: BaselineMode = "off"         # "off" / "each" / "batch"
    model_seed: int = 0



class ReinforceAgent:
    def __init__(
        self, 
        env: Game2048Env, 
        mlp_config: MLPConfig, 
        agent_config: ReinforceAgentConfig | None = None
    ):
        self.env = env
        self.mlp_config = mlp_config
        self.agent_config = agent_config or ReinforceAgentConfig()
        
        self.rng = np.random.default_rng(self.agent_config.model_seed)
        
        # logging
        self._logger = logging.getLogger(__name__ + ".ReinforceAgent")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())

        # get input_dim
        obs, info = self.env.reset(seed=0)
        x, action_mask = encode_observation(
            obs, self.mlp_config.use_onehot
        )
        input_dim = x.shape[0]
        n_actions = self.env.action_space.n
        
        if self.mlp_config.num_layers == 0:
            self.params = init_model_params_0layer(input_dim, n_actions, self.rng)
        else:
            raise NotImplementedError("multi layer not implement yet")


    def select_action(self, obs, rng: np.random.Generator) -> int | np.ndarray:
        '''
        Given observation, select action according to policy
        '''
        x, action_mask = encode_observation(obs, self.mlp_config.use_onehot)

        logits = forward_logits_0layer(self.params, x)

        probs = logits_to_probs(logits, action_mask)

        action = rng.choice(len(probs), p=probs)
        
        self._logger.debug(
            f"Selected action: {action}, probs: {probs}"
        )

        return action, probs


    def run_episode(self, env_seed: int, policy_seed: int) -> dict[str | Any]:
        '''
        Run one episode, return trajectory dict
        '''
        self._logger.verbose(f"Episode start: env_seed={env_seed}, policy_seed={policy_seed}")
        
        obs, info = self.env.reset(seed=env_seed)

        policy_rng = np.random.default_rng(policy_seed)

        obs_list: list[Any] = []
        action_list: list[int] = []
        reward_list: list[float] = []
        probs_list: list[np.ndarray] = []

        done = False
        total_reward = 0.0

        while not done:
            action, probs = self.select_action(obs, policy_rng)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            reward = float(reward)          # ensure reward is float

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            probs_list.append(probs)

            total_reward += reward
            obs = next_obs
            done = terminated or truncated

        trajectory = {
            "obs": obs_list,
            "actions": action_list,
            "rewards": reward_list,
            "probs": probs_list,
            "total_reward": total_reward,
        }
        
        self._logger.verbose(f"Episode finished, total_reward={total_reward:.3f}")
        self._logger.verbose(f"ENDGAME STATE\n" + self.env.render(mode="ansi"))
        
        return trajectory
