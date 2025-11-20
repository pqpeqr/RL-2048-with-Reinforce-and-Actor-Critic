import numpy as np

from env import Game2048Env

from collections.abc import Callable
from typing import Literal, Any
from dataclasses import dataclass
from collections.abc import Iterator


import logging

from MLP import *



BaselineMode = Literal["off", "each", "batch"]
OptimizerType = Literal["sgd", "adam"]

@dataclass
class ReinforceAgentConfig:
    gamma: float = 1                            # [0, 1]
    learning_rate: float = 1e-3                 # 
    baseline_mode: BaselineMode = "off"         # "off" / "each" / "batch"
    model_seed: int = 0
    normalize_advantage: bool = False           # whether to normalize advantages to mean 0, std 1
    # e.g., [1.0, 0.0] means training on bottom 50% reward episodes only
    # [3.0, 2.0, 1.0, 1.0] means weighting bottom 25% episodes 3x, next 25% 2x, other 50% 1x
    reward_rank_weights: list[float] | None = None
    
    optimizer: OptimizerType = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8



class ReinforceAgent:
    def __init__(
        self, 
        env: Game2048Env, 
        mlp_config: MLPConfig, 
        agent_config: ReinforceAgentConfig | None = None, 
        initial_params_path: str | None = None,
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
        x, action_mask = encode_observation(obs)
        input_dim = x.shape[0]
        n_actions = self.env.action_space.n
        
        # initialize model parameters
        if initial_params_path is None:
            self.params = init_model_params(input_dim,
                                            self.mlp_config.hidden_sizes,
                                            n_actions,
                                            self.rng,
                                            self.mlp_config.init_distribution,
                                            )
        else:
            self.params = load_model_params(initial_params_path)
            
        # Adam
        W_list: list[np.ndarray] = self.params["W"]
        b_list: list[np.ndarray] = self.params["b"]
        
        self._adam_m_W: list[np.ndarray] = [
            np.zeros_like(W, dtype=np.float32) for W in W_list
        ]
        self._adam_v_W: list[np.ndarray] = [
            np.zeros_like(W, dtype=np.float32) for W in W_list
        ]
        self._adam_m_B: list[np.ndarray] = [
            np.zeros_like(b, dtype=np.float32) for b in b_list
        ]
        self._adam_v_B: list[np.ndarray] = [
            np.zeros_like(b, dtype=np.float32) for b in b_list
        ]
        self._adam_t: int = 0
    
    
    def load_model(self, file_path: str | None = "params.npz") -> None:
        '''
        Load model parameters from npz file
        '''
        self.params = load_model_params(file_path)
    
    
    def save_model(self, file_path: str | None = "params.npz") -> None:
        '''
        Save model parameters to npz file
        '''
        save_model_params(self.params, file_path)


    def select_action(
        self, 
        obs, 
        rng: np.random.Generator, 
        action_fn: Callable[[Any, np.ndarray | None], int] | None = None,
    ) -> tuple[int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        '''
        Given observation, select action according to policy.
        If `action_fn` is provided, use it to select the final action,
        bypassing the model's action-selection logic.
        '''
        x, action_mask = encode_observation(obs)

        logits, activations, pre_activations = forward_logits(
            self.params,
            x,
            self.mlp_config.activation,
        )
        probs = logits_to_probs(logits, action_mask)
        
        self._logger.debug(
            f"Model logits: {logits}, probs: {probs}"
        )
        
        action: int | None = None

        if action_fn is not None:
            try:
                candidate = int(action_fn(self.env.state, action_mask))
            except Exception as e:
                self._logger.exception(
                    f"action_fn raised an exception: {e}. Falling back to policy."
                )
            else:
                # check candidate validity
                if not (0 <= candidate < len(probs)):
                    self._logger.warning(
                        f"action_fn returned out-of-range action {candidate}, "
                        "falling back to policy."
                    )
                elif action_mask is not None and not bool(action_mask[candidate]):
                    self._logger.warning(
                        f"action_fn returned masked-out action {candidate}, "
                        "falling back to policy."
                    )
                else:
                    action = candidate
                    self._logger.debug(
                        f"Selected action:{action} from action_fn."
                    )
        
        if action is None:
            action = int(rng.choice(len(probs), p=probs))

            self._logger.debug(
                f"Selected action:{action} from policy."
            )

        return action, probs, activations, pre_activations


    def run_episode(
        self, 
        env_seed: int, 
        policy_seed: int,
        action_gen: Callable[[Any, np.ndarray | None], int] | None = None,
        ) -> dict[str | Any]:
        '''
        Run one episode, return trajectory dict
        '''
        self._logger.verbose(f"Episode start: env_seed={env_seed}, policy_seed={policy_seed}")
        
        obs, info = self.env.reset(seed=env_seed)
        
        state = self.env.render(mode="ansi")

        policy_rng = np.random.default_rng(policy_seed)

        obs_list: list[Any] = []
        action_list: list[int] = []
        reward_list: list[float] = []
        probs_list: list[np.ndarray] = []
        activations_list: list[list[np.ndarray]] = []
        pre_activations_list: list[list[np.ndarray]] = []
        states_list: list[str] = []

        done = False
        total_reward = 0.0

        while not done:
            action, probs, activations, pre_activations = self.select_action(obs, policy_rng, action_gen)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            reward = float(reward)          # ensure reward is float

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            probs_list.append(probs)
            activations_list.append(activations)
            pre_activations_list.append(pre_activations)
            states_list.append(state)

            total_reward += reward
            obs = next_obs
            done = terminated or truncated
            state = self.env.render(mode="ansi")
        
        max_tile = self.env.max_tile_seen

        trajectory = {
            "obs": obs_list,
            "actions": action_list,
            "rewards": reward_list,
            "probs": probs_list,
            "activations": activations_list,
            "pre_activations": pre_activations_list,
            "total_reward": total_reward,
            "states": states_list,
            "max_tile": max_tile,
        }
        
        self._logger.verbose(f"Episode finished, total_reward={total_reward:.3f}, max_tile={max_tile}")
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
            advantages_list: list[np.ndarray] = [
                r.astype(np.float32) for r in returns_list
            ]

        elif mode == "each":        
            advantages_list = []
            for r in returns_list:
                baseline = float(r.mean())
                advantages = (r - baseline).astype(np.float32)
                advantages_list.append(advantages)

        elif mode == "batch":
            all_returns = np.concatenate(returns_list)
            baseline = float(all_returns.mean())
            advantages_list = [
                (r - baseline).astype(np.float32) for r in returns_list
            ]

        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
        
        # normalize advantages if enabled
        if self.agent_config.normalize_advantage:
            all_advantages = np.concatenate(advantages_list)
            mean = float(all_advantages.mean())
            std = float(all_advantages.std())
            
            eps = 1e-8 # to avoid division by zero
            if std < eps:
                std = eps

            advantages_list = [
                ((adv - mean) / std).astype(np.float32)
                for adv in advantages_list
            ]    
        
        return advantages_list


    def _policy_gradient_step(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        action: int,
        probs: np.ndarray,
        advantage: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Compute policy gradient for one time step
        '''
        # one_hot(a_t)
        one_hot = np.zeros_like(probs, dtype=np.float32)
        one_hot[action] = 1.0

        # gradient of logits:  d (log pi) / d z = one_hot - probs
        grad_logits = advantage * (one_hot - probs.astype(np.float32))

        # grad b and W
        dW_list, db_list = self._backpropagation(
            activations=activations,
            pre_activations=pre_activations,
            grad_logits=grad_logits,
        )

        return dW_list, db_list


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
        
        episode_rank_weights = self._compute_episode_rank_weights(returns_list)

        W_list: list[np.ndarray] = self.params["W"]
        b_list: list[np.ndarray] = self.params["b"]

        # initialize gradients
        grad_W_list: list[np.ndarray] = [
            np.zeros_like(W_l, dtype=np.float32) for W_l in W_list
        ]
        grad_b_list: list[np.ndarray] = [
            np.zeros_like(b_l, dtype=np.float32) for b_l in b_list
        ]
        
        n_traj = len(trajectories)
        if n_traj == 0:
            return

        # accumulate gradients
        for epi_idx, (traj, advantages) in enumerate(zip(trajectories, advantages_list)):
            obs_list = traj["obs"]
            action_list = traj["actions"]
            probs_list = traj["probs"]
            activations_list = traj["activations"]
            pre_activations_list = traj["pre_activations"]
            
            T = len(obs_list)
            if T == 0:
                continue

            # average over all time steps then over all trajectories
            base_episode_weight = 1.0 / (T * n_traj)
            rank_weight = float(episode_rank_weights[epi_idx])  # e.g. 3.0, 2.0, 1.0, 0.0 ...
            episode_weight = base_episode_weight * rank_weight
            
            for action, adv, probs, activations, pre_activations in zip(
                action_list, advantages, probs_list, activations_list, pre_activations_list
            ):
                dW_list, db_list = self._policy_gradient_step(
                    activations=activations,
                    pre_activations=pre_activations,
                    action=action,
                    probs=probs,
                    advantage=float(adv),
                )

                for l in range(len(grad_W_list)):
                    grad_W_list[l] += episode_weight * dW_list[l]
                    grad_b_list[l] += episode_weight * db_list[l]
                    
    


        # logging for gradient norms
        if self._logger.isEnabledFor(logging.INFO):
            batch_grad_W_norms = [np.linalg.norm(gW) for gW in grad_W_list]
            batch_grad_b_norms = [np.linalg.norm(gb) for gb in grad_b_list]
            
            batch_grad_W_norms_str = ", ".join(f"{n:.6f}" for n in batch_grad_W_norms)
            batch_grad_b_norms_str = ", ".join(f"{n:.6f}" for n in batch_grad_b_norms)

            self._logger.info(
                f"Step grad_W norms: [{batch_grad_W_norms_str}], "
                f"grad_b norms: [{batch_grad_b_norms_str}]"
            )


        # update parameters (gradient ascent)
        if self.agent_config.optimizer == "sgd":
            lr = self.agent_config.learning_rate
            for l in range(len(W_list)):
                self.params["W"][l] = W_list[l] + lr * grad_W_list[l]
                self.params["b"][l] = b_list[l] + lr * grad_b_list[l]

        elif self.agent_config.optimizer == "adam":
            self._adam_update(grad_W_list, grad_b_list)

        else:
            raise ValueError(f"Unknown optimizer: {self.agent_config.optimizer}")
            
            
            
    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function
        """
        if self.mlp_config.activation == "Sigmoid":
            # σ(z) = 1 / (1 + exp(-z))
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        elif self.mlp_config.activation == "ReLU":
            # ReLU'(z) = 1 if z > 0 else 0
            return (z > 0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported activation: {self.mlp_config.activation}")
        
        
    def _backpropagation(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        grad_logits: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Backpropagation through MLP to compute gradients of parameters
        """
                
        Ws: list[np.ndarray] = self.params["W"]
        bs: list[np.ndarray] = self.params["b"]
        num_layers = len(Ws)

        # init gradients
        grad_W_list: list[np.ndarray] = [
            np.zeros_like(W, dtype=np.float32) for W in Ws
        ]
        grad_b_list: list[np.ndarray] = [
            np.zeros_like(b, dtype=np.float32) for b in bs
        ]

        delta = grad_logits.astype(np.float32)

        for l in reversed(range(num_layers)):
            a_prev = activations[l]
            grad_W_list[l] = np.outer(a_prev, delta)
            grad_b_list[l] = delta

            if l > 0:
                W_l = Ws[l]
                da_prev = delta @ W_l.T
                z_prev = pre_activations[l - 1]
                dz_prev = da_prev * self._activation_derivative(
                    z_prev,
                )
                delta = dz_prev

        return grad_W_list, grad_b_list


    def _compute_episode_rank_weights(
        self,
        returns_list: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute episode weights based on reward ranks and user-configured weights.
        """
        weights_conf = self.agent_config.reward_rank_weights
        
        n_traj = len(returns_list)
        if n_traj == 0:
            return np.array([], dtype=np.float32)
        
        # if no weights configured, return all ones
        if weights_conf is None or len(weights_conf) == 0:
            return np.ones(n_traj, dtype=np.float32)
        
        weights_conf = np.asarray(weights_conf, dtype=np.float32)
        num_bins = len(weights_conf)
        
        total_returns = np.array(
            [float(r.sum()) for r in returns_list],
            dtype=np.float32,
        )
        
        sorted_indices = np.argsort(total_returns)  # shape (n_traj,)
        
        episode_weights = np.zeros(n_traj, dtype=np.float32)
        
        for rank, epi_idx in enumerate(sorted_indices):
            percent_rank = (rank + 0.5) / n_traj  # e.g., rank=0 ~ 0.5/n, rank=n-1 ~ (n-0.5)/n
            bin_idx = int(percent_rank * num_bins)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            episode_weights[epi_idx] = weights_conf[bin_idx]
        
        return episode_weights


    def _adam_update(
        self,
        grad_W_list: list[np.ndarray],
        grad_b_list: list[np.ndarray],
    ) -> None:
        """
        Update model parameters using Adam optimizer
        """
        lr = self.agent_config.learning_rate
        beta1 = self.agent_config.adam_beta1
        beta2 = self.agent_config.adam_beta2
        eps = self.agent_config.adam_eps

        self._adam_t += 1
        t = self._adam_t

        for l in range(len(self.params["W"])):
            gW = grad_W_list[l]
            gb = grad_b_list[l]

            self._adam_m_W[l] = beta1 * self._adam_m_W[l] + (1.0 - beta1) * gW
            self._adam_m_B[l] = beta1 * self._adam_m_B[l] + (1.0 - beta1) * gb

            self._adam_v_W[l] = beta2 * self._adam_v_W[l] + (1.0 - beta2) * (gW * gW)
            self._adam_v_B[l] = beta2 * self._adam_v_B[l] + (1.0 - beta2) * (gb * gb)

            # bias correction
            mW_hat = self._adam_m_W[l] / (1.0 - beta1 ** t)
            mB_hat = self._adam_m_B[l] / (1.0 - beta1 ** t)
            vW_hat = self._adam_v_W[l] / (1.0 - beta2 ** t)
            vB_hat = self._adam_v_B[l] / (1.0 - beta2 ** t)

            self.params["W"][l] = self.params["W"][l] + lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.params["b"][l] = self.params["b"][l] + lr * mB_hat / (np.sqrt(vB_hat) + eps)
