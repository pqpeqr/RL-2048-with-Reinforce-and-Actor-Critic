import numpy as np

from env import Game2048Env
from typing import Any

from MLP import (
    MLPConfig,
    encode_observation,
    init_model_params_0layer,
    forward_logits_0layer,
    logits_to_probs,
)


class ReinforceAgent:
    def __init__(self, env: Game2048Env, mlp_config: MLPConfig, model_seed: int = 0):
        self.env = env
        self.config = mlp_config
        self.rng = np.random.default_rng(model_seed)

        # get input_dim
        obs, info = self.env.reset(seed=0)
        x, action_mask = encode_observation(
            obs, self.config.use_onehot
        )
        input_dim = x.shape[0]
        n_actions = self.env.action_space.n

        if self.config.num_layers == 0:
            self.params = init_model_params_0layer(input_dim, n_actions, self.rng)
        else:
            pass


    def select_action(self, obs, rng: np.random.Generator) -> int | np.ndarray:
        x, action_mask = encode_observation(obs, self.config.use_onehot)

        logits = forward_logits_0layer(self.params, x)

        probs = logits_to_probs(logits, action_mask)

        action = rng.choice(len(probs), p=probs)

        return action, probs


    def run_episode(self, env_seed: int, policy_seed: int) -> dict[str | Any]:
        obs, info = self.env.reset(seed=env_seed)

        policy_rng = np.random.default_rng(policy_seed)

        obs_list = []
        action_list = []
        reward_list = []
        probs_list = []

        done = False
        total_reward = 0.0

        while not done:
            action, probs = self.select_action(obs, policy_rng)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

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
        return trajectory
