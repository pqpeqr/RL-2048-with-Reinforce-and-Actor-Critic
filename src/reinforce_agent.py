import numpy as np

from .env import Game2048Env

from collections.abc import Callable
from typing import Literal, Any
from dataclasses import dataclass
from collections.abc import Iterator


import logging
import src.utils.logging_ext as logging_ext      # add VERBOSE level to logging

from .MLP import *




BaselineMode = Literal["off", "each", "batch", "batch_norm"]
OptimizerType = Literal["sgd", "adam"]
CriticLossType = Literal["mse", "huber"]


@dataclass
class ReinforceAgentConfig:
    gamma: float = 1                            # [0, 1]
    learning_rate: float = 1e-3                 # 
    baseline_mode: BaselineMode = "off"         # "off" / "each" / "batch" / "batch_norm"
    model_seed: int = 0
    # e.g., [1.0, 0.0] means training on bottom 50% reward episodes only
    # [3.0, 2.0, 1.0, 1.0] means weighting bottom 25% episodes 3x, next 25% 2x, other 50% 1x
    reward_rank_weights: list[float] | None = None
    
    optimizer: OptimizerType = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    augmentation: bool = False                  # whether to use data augmentation (symmetries)
    
    use_critic: bool = False                    # Actor-Critic switch
    critic_learning_rate: float = 1e-3          # Critic learning rate
    
    max_grad_norm: 1.0                          # global gradient clipping norm
    
    critic_loss_type: CriticLossType = "mse"    # "mse" or "huber"
    huber_delta: float = 1.0                    # delta for Huber loss (if used)



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
                                            self.mlp_config.last_init_normal
                                            )
        else:
            self.params = load_model_params(initial_params_path)
            
        # Adam
        self._init_adam(self.params, prefix="actor")
        self._adam_t: int = 0
        
        # critic
        self.critic_params = None
        
        if self.agent_config.use_critic:
            self.critic_params = init_model_params(
                input_dim,
                self.mlp_config.hidden_sizes, 
                1,
                self.rng,
                self.mlp_config.init_distribution,
                self.mlp_config.last_init_normal
            )
            # Adam for critic
            self._init_adam(self.critic_params, prefix="critic")
            self._adam_t_c: int = 0
    
    
    def load_model(self, file_path: str | None = "params.npz") -> None:
        '''
        Load model parameters from npz file
        '''
        self.params = load_model_params(file_path)
        
        self.mlp_config.hidden_sizes = [W.shape[1] for W in self.params["W"][:-1]]
        
        self._logger.info(f"Model parameters loaded from {file_path}")
    
    
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
        use_greedy: bool = False,
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
            if use_greedy:
                if action_mask is not None:
                    probs = probs * action_mask
                action = int(np.argmax(probs))
                self._logger.debug(
                    f"Selected action (greedy argmax): {action}"
                )
            else:
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
        use_greedy: bool = False,   # if True, select action with argmax
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
        states_list: list[str] = []

        done = False
        total_reward = 0.0

        while not done:
            action, _, _, _ = self.select_action(obs, policy_rng, action_gen, use_greedy=use_greedy)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            reward = float(reward)          # ensure reward is float

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
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
        episode_rank_weights: np.ndarray,
    ) -> list[np.ndarray]:
        """
        A_t = G_t - baseline (depending on baseline_mode). 
        Compute advantages based on baseline_mode:
        - "off"  : advantage = returns
        - "each" : baseline by episode, advantage = returns - mean(returns)
        - "batch": baseline by batch, advantage = returns - mean(all returns in batch)
        - "batch_norm": baseline by batch, normalized advantage = (returns - mean) / std
        """
        mode = self.agent_config.baseline_mode

        if mode == "off":
            return [
                r.astype(np.float32) for r in returns_list
            ]

        elif mode == "each":        
            advantages_list = []
            for r in returns_list:
                baseline = float(r.mean())
                advantages = (r - baseline).astype(np.float32)
                advantages_list.append(advantages)
            return advantages_list

        all_values = np.concatenate(returns_list)
        weights_expanded = []
        for i, r in enumerate(returns_list):
            w = episode_rank_weights[i]
            weights_expanded.append(np.full(len(r), w))

        all_weights = np.concatenate(weights_expanded)
        
        mean, std = self._compute_weighted_stats(all_values, all_weights)
        
        if mode == "batch":
            return [
                (r - mean).astype(np.float32) for r in returns_list
            ]
        elif mode == "batch_norm":
            if std < 1e-8:
                std = 1e-8
            return [
                ((r - mean) / std).astype(np.float32) for r in returns_list
            ]
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")


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
            params=self.params,
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
        total_reward_list : list[float] = []
        for traj in trajectories:
            total_reward_list.append(traj["total_reward"])
            if not self.agent_config.use_critic:
                returns_list.append(self.compute_returns(traj["rewards"]))
        
        episode_rank_weights = self._compute_episode_rank_weights(total_reward_list)

        
        if not self.agent_config.use_critic:
            # A_t = G_t - baseline (depending on baseline_mode)
            advantages_list = self._compute_advantages(returns_list, episode_rank_weights)
        else:
            advantages_list = [None] * len(trajectories)

        # augmentation
        if self.agent_config.augmentation:
            trajectories, advantages_list, episode_rank_weights = self._augment_trajectories(
                trajectories, 
                advantages_list, 
                episode_rank_weights
            )
            
        # initialize gradients
        grad_W_list: list[np.ndarray] = [
            np.zeros_like(W_l, dtype=np.float32) for W_l in self.params["W"]
        ]
        grad_b_list: list[np.ndarray] = [
            np.zeros_like(b_l, dtype=np.float32) for b_l in self.params["b"]
        ]
        
        if self.agent_config.use_critic and self.critic_params is not None:
            grad_W_c_list = [np.zeros_like(W) for W in self.critic_params["W"]]
            grad_b_c_list = [np.zeros_like(b) for b in self.critic_params["b"]]
        
        n_traj = len(trajectories)
        if n_traj == 0:
            return

        # update critic and compute TD errors
        if self.agent_config.use_critic and self.critic_params is not None:
            td_errors_list: list[np.ndarray] = []
            
            for traj, episode_rank_weight in zip(trajectories, episode_rank_weights):
                obs_list = traj["obs"]
                reward_list = traj["rewards"]
                T = len(obs_list)
                if T == 0:
                    td_errors_list.append(np.zeros(0, dtype=np.float32))
                    continue
                
                x_list = []
                for o in obs_list:
                    x, _ = encode_observation(o)
                    x_list.append(x)
                X_batch = np.array(x_list)  # (T, input_dim)
                
                reward_batch = np.array(reward_list, dtype=np.float32)  # (T,)
                # move one step forward for next state values
                # padding the last state with itself, will be masked out by done mask
                X_next_batch = np.concatenate([X_batch[1:], X_batch[-1:]], axis=0)
                
                v_logits, v_acts, v_pre_acts = forward_logits(
                    self.critic_params, 
                    X_batch, 
                    self.mlp_config.activation
                )
                v_curr = v_logits.flatten()         # Shape: (T, 1) -> (T,)
                
                v_next_logits, _, _ = forward_logits(
                    self.critic_params, 
                    X_next_batch, 
                    self.mlp_config.activation
                )
                v_next = v_next_logits.flatten()    # Shape: (T, 1) -> (T,)
                
                mask_done = np.ones(T, dtype=np.float32)
                mask_done[-1] = 0.0
                
                gamma = self.agent_config.gamma
                td_targets = reward_batch + gamma * v_next * mask_done
                
                td_errors = td_targets - v_curr
                
                td_errors_list.append(td_errors.astype(np.float32))
                
                # compute critic gradients
                # loos = 0.5 * (v_curr - td_targets)^2
                # dloss / dv_logits = v_curr - td_targets
                grad_logits_c = self._get_grad_logits_critic(v_curr, td_targets)
                
                # backpropagate through time for critic
                for t in range(T):
                    v_acts_t = [layer[t] for layer in v_acts]
                    v_pre_t  = [layer[t] for layer in v_pre_acts]
                    grad_logits_c_t = grad_logits_c[t]

                    dW_c_t, db_c_t = self._backpropagation(
                        self.critic_params, 
                        v_acts_t, 
                        v_pre_t, 
                        grad_logits_c_t
                    )
                    
                    base_weight = 1.0 / (T * n_traj)
                    c_weight = base_weight * float(episode_rank_weight)

                    for l in range(len(grad_W_c_list)):
                        grad_W_c_list[l] += c_weight * dW_c_t[l]
                        grad_b_c_list[l] += c_weight * db_c_t[l]
                
                # logging
                td_loss = 0.5 * float(np.mean(td_errors ** 2))
                td_abs_mean = float(np.mean(np.abs(td_errors)))
                td_abs_max = float(np.max(np.abs(td_errors)))
                v_mean = float(np.mean(v_curr))
                v_std = float(np.std(v_curr) + 1e-8)
                # Monte-Carlo return
                returns_mc = self.compute_returns(reward_list)
                if returns_mc.std() > 1e-8 and v_std > 1e-8:
                    corr_v_ret = float(np.corrcoef(v_curr, returns_mc)[0, 1])
                else:
                    corr_v_ret = 0.0
                self._logger.verbose(
                    f"[Critic] td_loss={td_loss:.6f}, "
                    f"|δ|_mean={td_abs_mean:.6f}, "
                    f"|δ|_max={td_abs_max:.6f}, "
                    f"V_mean={v_mean:.6f}, "
                    f"V_std={v_std:.6f}, "
                    f"corr(V, G)={corr_v_ret:.3f}"
                )
                
            advantages_list = self._compute_advantages(
                td_errors_list,
                episode_rank_weights,
            )
                
        
        # actor policy gradient update
        for traj, advantages, episode_rank_weight in zip(
            trajectories, 
            advantages_list, 
            episode_rank_weights,
        ):
            obs_list = traj["obs"]
            action_list = traj["actions"]
            
            T = len(obs_list)
            if T == 0:
                continue
            
            x_list = []
            mask_list = []
            
            for o in obs_list:
                x, mask = encode_observation(o) 
                x_list.append(x)
                mask_list.append(mask)
            
            X_batch = np.array(x_list)        # Shape: (T, Input_Dim)
            mask_batch = np.array(mask_list)  # Shape: (T, N_Actions)
            
            logits_batch, activations_batch, pre_activations_batch = forward_logits(
                self.params,
                X_batch,
                self.mlp_config.activation,
            )
            probs_batch = logits_to_probs(logits_batch, mask_batch)

            # average over all time steps then over all trajectories
            base_episode_weight = 1.0 / (T * n_traj)
            weight = base_episode_weight * float(episode_rank_weight)
            
            for t in range(T):
                action = action_list[t]
                adv = float(advantages[t])
                
                probs = probs_batch[t]
                
                act_t = [layer[t] for layer in activations_batch]
                pre_act_t = [layer[t] for layer in pre_activations_batch]
                
                dW_t, db_t = self._policy_gradient_step(
                    activations=act_t,
                    pre_activations=pre_act_t,
                    action=action,
                    probs=probs,
                    advantage=adv
                )
                
                for l in range(len(grad_W_list)):
                    grad_W_list[l] += weight * dW_t[l]
                    grad_b_list[l] += weight * db_t[l]
        
        
        # global gradient clipping
        actor_grad_norm = self.clip_grads_global_norm(grad_W_list, grad_b_list)
        if self.agent_config.use_critic and self.critic_params is not None:
            critic_grad_norm = self.clip_grads_global_norm(grad_W_c_list, grad_b_c_list)
        
        
        # update parameters (gradient ascent)
        if self.agent_config.optimizer == "sgd":
            lr = self.agent_config.learning_rate
            for l in range(len(self.params["W"])):
                self.params["W"][l] += lr * grad_W_list[l]
                self.params["b"][l] += lr * grad_b_list[l]
            # critic
            lr_c = self.agent_config.critic_learning_rate
            if self.agent_config.use_critic and self.critic_params is not None:
                for l in range(len(self.critic_params["W"])):
                    self.critic_params["W"][l] -= lr_c * grad_W_c_list[l]
                    self.critic_params["b"][l] -= lr_c * grad_b_c_list[l]
        elif self.agent_config.optimizer == "adam":
            self._adam_update(grad_W_list, grad_b_list)
            # critic
            if self.agent_config.use_critic and self.critic_params is not None:
                self._adam_update(grad_W_c_list, grad_b_c_list, prefix="critic")
        else:
            raise ValueError(f"Unknown optimizer: {self.agent_config.optimizer}")
        

        # logging for gradient norms
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                f"Global Grad Norms: Actor: {actor_grad_norm:.4f}"
            )
            if self.agent_config.use_critic and self.critic_params is not None:
                self._logger.info(
                    f"Global Grad Norms: Critic: {critic_grad_norm:.4f}"
                )
            
            # advantages info
            all_adv_list = [a for a in advantages_list if a is not None and len(a) > 0]
            if len(all_adv_list) > 0:
                all_adv = np.concatenate(all_adv_list).astype(np.float32)
                
                adv_mean = float(all_adv.mean())
                adv_std  = float(all_adv.std())
                adv_min  = float(all_adv.min())
                adv_max  = float(all_adv.max())

                # prob of positive/negative advantages
                num_pos = int((all_adv > 0).sum())
                num_neg = int((all_adv < 0).sum())
                total   = len(all_adv)
                
                # outliers
                k = min(5, total)
                top_idx = np.argsort(-np.abs(all_adv))[:k]
                top_vals = ", ".join(f"{all_adv[i]:.4f}" for i in top_idx)

                self._logger.info(
                    "Advantages stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, "
                    "pos=%d, neg=%d, total=%d, top|A|=%s",
                    adv_mean, adv_std, adv_min, adv_max,
                    num_pos, num_neg, total, top_vals,
                )



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
        params: dict[str, Any],
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        grad_logits: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Backpropagation through MLP to compute gradients of parameters
        """
                
        Ws = params["W"]
        bs = params["b"]
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
        total_reward_list: list[float],
    ) -> np.ndarray:
        """
        Compute episode weights based on reward ranks and user-configured weights.
        """
        weights_conf = self.agent_config.reward_rank_weights
        
        n_traj = len(total_reward_list)
        if n_traj == 0:
            return np.array([], dtype=np.float32)
        
        # if no weights configured, return all ones
        if weights_conf is None or len(weights_conf) == 0:
            return np.ones(n_traj, dtype=np.float32)
        
        weights_conf = np.asarray(weights_conf, dtype=np.float32)
        num_bins = len(weights_conf)
        
        sorted_indices = np.argsort(total_reward_list)
        
        episode_weights = np.zeros(n_traj, dtype=np.float32)
        
        for rank, epi_idx in enumerate(sorted_indices):
            percent_rank = (rank + 0.5) / n_traj  # e.g., rank=0 ~ 0.5/n, rank=n-1 ~ (n-0.5)/n
            bin_idx = int(percent_rank * num_bins)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            episode_weights[epi_idx] = weights_conf[bin_idx]
            
        mean_w = np.mean(episode_weights)
        if mean_w > 1e-8:
            episode_weights = episode_weights / mean_w
        
        return episode_weights


    def _adam_update(
        self,
        grad_W_list: list[np.ndarray],
        grad_b_list: list[np.ndarray],
        prefix="actor"
    ) -> None:
        """
        Update model parameters using Adam optimizer
        """
        if prefix == "actor":
            params = self.params
            m_W, v_W = self._adam_m_W, self._adam_v_W
            m_B, v_B = self._adam_m_B, self._adam_v_B
            lr = self.agent_config.learning_rate
            
            self._adam_t += 1
            t = self._adam_t
            
            update_sign = 1.0 # gradient ascent
        else: # critic
            params = self.critic_params
            m_W, v_W = self._adam_m_W_c, self._adam_v_W_c
            m_B, v_B = self._adam_m_B_c, self._adam_v_B_c
            lr = self.agent_config.critic_learning_rate
            
            self._adam_t_c += 1
            t = self._adam_t_c

            update_sign = -1.0
        
        beta1 = self.agent_config.adam_beta1
        beta2 = self.agent_config.adam_beta2
        eps = 1e-8

        for l in range(len(params["W"])):
            gW = grad_W_list[l]
            gb = grad_b_list[l]

            m_W[l] = beta1 * m_W[l] + (1.0 - beta1) * gW
            m_B[l] = beta1 * m_B[l] + (1.0 - beta1) * gb

            v_W[l] = beta2 * v_W[l] + (1.0 - beta2) * (gW * gW)
            v_B[l] = beta2 * v_B[l] + (1.0 - beta2) * (gb * gb)

            # bias correction
            mW_hat = m_W[l] / (1.0 - beta1 ** t)
            mB_hat = m_B[l] / (1.0 - beta1 ** t)
            vW_hat = v_W[l] / (1.0 - beta2 ** t)
            vB_hat = v_B[l] / (1.0 - beta2 ** t)

            params["W"][l] = params["W"][l] + update_sign * lr * mW_hat / (np.sqrt(vW_hat) + eps)
            params["b"][l] = params["b"][l] + update_sign * lr * mB_hat / (np.sqrt(vB_hat) + eps)


    def _augment_trajectories(
        self, 
        trajectories: list[dict[str, Any]],
        advantages_list: list[np.ndarray],
        episode_rank_weights: np.ndarray,
        ) -> tuple[list[dict[str, Any]], list[np.ndarray], np.ndarray]:
        """
        Perform data augmentation on trajectories using environment symmetries.
        Return a new list of augmented trajectories.
        """
        augmented_trajectories = []

        for traj in trajectories:
            orig_obs = traj["obs"] 
            orig_actions = traj["actions"]
            orig_reward = traj["rewards"]
            
            aug_data = [ {"obs": [], "actions": [], "rewards": []} for _ in range(8) ]
            
            for t in range(len(orig_obs)):
                syms = self.env.get_symmetries(orig_obs[t], orig_actions[t])
                s_reward = orig_reward[t]
                
                for i in range(8):
                    s_board, s_action = syms[i]
                    aug_data[i]["obs"].append(s_board)
                    aug_data[i]["actions"].append(s_action)
                    aug_data[i]["rewards"].append(s_reward)

            augmented_trajectories.extend(aug_data)
            
        augmented_advantages_list: list[np.ndarray] = [x for x in advantages_list for _ in range(8)]
        
        augmented_episode_rank_weights = np.repeat(episode_rank_weights, 8)
            
        return augmented_trajectories, augmented_advantages_list, augmented_episode_rank_weights


    def _init_adam(self, params, prefix="actor"):
        """
        Initialize Adam optimizer states
        """
        W_list = params["W"]
        b_list = params["b"]
        
        m_W = [np.zeros_like(W, dtype=np.float32) for W in W_list]
        v_W = [np.zeros_like(W, dtype=np.float32) for W in W_list]
        m_B = [np.zeros_like(b, dtype=np.float32) for b in b_list]
        v_B = [np.zeros_like(b, dtype=np.float32) for b in b_list]
        
        if prefix == "actor":
            self._adam_m_W = m_W
            self._adam_v_W = v_W
            self._adam_m_B = m_B
            self._adam_v_B = v_B
        elif prefix == "critic":
            self._adam_m_W_c = m_W
            self._adam_v_W_c = v_W
            self._adam_m_B_c = m_B
            self._adam_v_B_c = v_B
            
            
    def clip_grads_global_norm(self, grad_W_list, grad_b_list):
        """
        calculate l2 norm of all gradients and clip them if exceed max_norm
        """
        max_norm = self.agent_config.max_grad_norm
        
        total_norm_sq = 0.0
        for gW in grad_W_list:
            total_norm_sq += np.linalg.norm(gW)**2
        for gb in grad_b_list:
            total_norm_sq += np.linalg.norm(gb)**2

        total_norm = np.sqrt(total_norm_sq)
        
        clip_coef = max_norm / max(total_norm, 1e-8)

        if clip_coef < 1.0:
            for i in range(len(grad_W_list)):
                grad_W_list[i] *= clip_coef
            for i in range(len(grad_b_list)):
                grad_b_list[i] *= clip_coef
            self._logger.verbose(
                f"Clipping gradients: total_norm={total_norm:.4f} > max_norm={max_norm:.4f}, "
                f"clip_coef={clip_coef:.6f}"
            )
                
        return total_norm   # return original norm for logging purposes
    
    
    def _compute_weighted_stats(
        self, 
        values: np.ndarray, 
        weights: np.ndarray
    ) -> tuple[float, float]:
        """
        Helper to compute weighted mean and std.
        Returns (mean, std).
        """
        sum_weights = np.sum(weights)
        if sum_weights < 1e-8:
            return 0.0, 1.0
        
        mean = np.sum(values * weights) / sum_weights
        var = np.sum(weights * (values - mean)**2) / sum_weights
        std = np.sqrt(var)
        
        return mean, std
    
    
    def _get_grad_logits_critic(self, v_curr: np.ndarray, td_targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of critic logits
        """
        diff = v_curr - td_targets
        
        if self.agent_config.critic_loss_type == "mse":
            # loss = 0.5 * (v_curr - td_targets)^2
            # dloss / dv_logits = v_curr - td_targets
            grad_logits_c = diff.astype(np.float32).reshape(-1, 1)
        elif self.agent_config.critic_loss_type == "huber":
            # Huber loss
            delta = self.agent_config.huber_delta
            abs_diff = np.abs(diff)
            
            # Gradients:
            # if |x| <= delta: x
            # if |x| > delta:  delta * sign(x)
            grad_logits_c = np.where(
                abs_diff <= delta,
                diff,
                delta * np.sign(diff)
            ).astype(np.float32).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown critic loss type: {self.agent_config.critic_loss_type}")

        return grad_logits_c
