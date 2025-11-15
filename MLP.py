import numpy as np
from typing import Literal

from dataclasses import dataclass
from typing import Any

ActivationMode = Literal["Sigmoid", "ReLU", ]


@dataclass
class MLPConfig:
    use_onehot: bool = False
    num_layers: int = 0
    activation: ActivationMode = "Sigmoid"
    
    
def encode_observation(obs, use_onehot: bool) -> tuple[np.ndarray, np.ndarray | None]:
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


def init_model_params_0layer(input_dim: int, n_actions: int) -> dict[str, Any]:
    params = {}
    W = np.random.standard_normal((input_dim, n_actions)).astype(np.float32) * 0.01
    b = np.zeros((n_actions,), dtype=np.float32)

    params["W"] = W
    params["b"] = b
    return params


def forward_logits_0layer(params, x) -> np.ndarray:
    W = params["W"]
    b = params["b"]
    logits = x @ W + b
    return logits


def logits_to_probs(logits, action_mask = None) -> np.ndarray:
    if action_mask is not None:
        mask = action_mask.astype(bool)
        logits = np.where(mask, logits, -1e9)

    # softmax with numerically stable
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    probs = exps / np.sum(exps)
    return probs
