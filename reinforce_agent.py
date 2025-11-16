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


    def compute_returns(self, rewards: list[float]) -> np.ndarray:
        '''
        G_t = sum_{k=t}^{T-1} gamma^(k-t) R_{k+1}
        G_{T-1} = R_T
        G_{T-2} = R_{T-1} + gamma * G_T
        G_{T-3} = R_{T-2} + gamma * G_{T-1} + gamma^2 * G_T
        ...
        '''
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)

        G = 0.0
        gamma = self.agent_config.gamma

        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        return returns
    
    
    def _compute_advantages(
        self,
        returns_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        A_t = G_t - baseline (depending on baseline_mode). 
        Compute advantages based on baseline_mode:
        - "off"  : advantage = returns
        - "each" : baseline by episode, advantage = returns - mean(returns)
        - "batch": baseline by batch, advantage = returns - mean(all returns in batch)
        """
        mode = self.agent_config.baseline_mode

        if mode == "off":
            return [r.astype(np.float32) for r in returns_list]

        elif mode == "each":
            advantages_list: list[np.ndarray] = []
            for r in returns_list:
                baseline = float(r.mean())
                advantages = (r - baseline).astype(np.float32)
                advantages_list.append(advantages)
            return advantages_list

        elif mode == "batch":
            all_returns = np.concatenate(returns_list)
            baseline = float(all_returns.mean())

            advantages_list = [(r - baseline).astype(np.float32) for r in returns_list]
            return advantages_list

        else:
            raise ValueError(f"Unknown baseline mode: {mode}")


    def _policy_gradient_step(
        self,
        x: np.ndarray,
        action: int,
        probs: np.ndarray,
        weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute policy gradient for one time step
        '''
        # one_hot(a_t)
        one_hot = np.zeros_like(probs, dtype=np.float32)
        one_hot[action] = 1.0

        # gradient of logits:  d (log pi) / d z = one_hot - probs
        grad_logits = weight * (one_hot - probs.astype(np.float32))

        # grad b and W
        grad_b = grad_logits
        grad_W = np.outer(x.astype(np.float32), grad_logits)

        return grad_W, grad_b


    def update_batch(self, trajectories: list[dict[str, Any]]) -> None:
        '''
        Update model parameters using a batch of episode trajectories
        '''
        # 
        returns_list: list[np.ndarray] = []
        for traj in trajectories:
            rewards = traj["rewards"]
            returns = self.compute_returns(rewards)
            returns_list.append(returns)

        # A_t = G_t - baseline (depending on baseline_mode)
        advantages_list = self._compute_advantages(returns_list)

        # init W and b
        W = self.params["W"]
        b = self.params["b"]

        grad_W = np.zeros_like(W, dtype=np.float32)
        grad_b = np.zeros_like(b, dtype=np.float32)

        # accumulate gradients
        for traj, advantages in zip(trajectories, advantages_list):
            obs_list = traj["obs"]
            action_list = traj["actions"]
            probs_list = traj["probs"]

            for obs, action, adv, probs in zip(
                obs_list, action_list, advantages, probs_list
            ):
                x, _ = encode_observation(obs, self.mlp_config.use_onehot)

                dW, db = self._policy_gradient_step(
                    x=x,
                    action=action,
                    probs=probs,
                    weight=float(adv),
                )

                grad_W += dW
                grad_b += db

        # update parameters (gradient ascent)
        lr = self.agent_config.learning_rate
        self.params["W"] = W + lr * grad_W
        self.params["b"] = b + lr * grad_b
