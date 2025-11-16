import numpy as np
from typing import Literal

from dataclasses import dataclass
from typing import Any

ActivationMode = Literal["Sigmoid", "ReLU", ]


@dataclass
class MLPConfig:
    use_onehot: bool = False                # T / F
    num_layers: int = 0                     # 0 / 1 / 2/ 3
    activation: ActivationMode = "Sigmoid"  # "Sigmoid" / "ReLU"
    
    
def encode_observation(obs, use_onehot: bool) -> tuple[np.ndarray, np.ndarray | None]:
    '''
    Encode observation into input vector x for MLP
    Input:
        obs: observation from env
             dict of "board" and "action_mask" if action mask is used
             "board" only otherwise
        use_onehot: whether to use onehot encoding
    Output:
        x: input vector for MLP
        action_mask: action mask if available, else None
    '''
    if isinstance(obs, dict):
        board = obs["board"]
        action_mask = obs["action_mask"]
    else:
        board = obs
        action_mask = None

    if use_onehot:
        # TODO
        raise NotImplementedError("onehot not implement yet")
    else:
        x = board.astype(np.float32).flatten()

    return x, action_mask


def init_model_params_0layer(
    input_dim: int, 
    n_actions: int, 
    rng: np.random.Generator
) -> dict[str, Any]:
    '''
    Initialize model parameters for 0-layer MLP (linear model)
    '''
    params = {}
    W = rng.standard_normal((input_dim, n_actions), dtype=np.float32) * 0.01
    b = np.zeros((n_actions,), dtype=np.float32)

    params["W"] = W
    params["b"] = b
    return params


def forward_logits_0layer(params, x) -> np.ndarray:
    '''
    Forward pass for 0-layer MLP (linear model)
    '''
    W = params["W"]
    b = params["b"]
    logits = x @ W + b
    return logits


def logits_to_probs(logits, action_mask = None) -> np.ndarray:
    '''
    Convert logits to action probabilities using softmax
    '''
    if action_mask is not None:
        mask = action_mask.astype(bool)
        logits = np.where(mask, logits, -1e9)

    # softmax with numerically stable
    # The Log-Sum-Exp Trick
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    probs = exps / np.sum(exps)
    return probs
