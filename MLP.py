import numpy as np
from typing import Literal

from dataclasses import dataclass, field
from typing import Any

ActivationMode = Literal["Sigmoid", "ReLU", ]


@dataclass
class MLPConfig:
    use_onehot: bool = False                                # T / F
    hidden_sizes: list[int] = field(default_factory=list)   # list of hidden layer sizes
    activation: ActivationMode = "Sigmoid"                  # "Sigmoid" / "ReLU"
    
    @property
    def num_layers(self) -> int:
        return len(self.hidden_sizes)
    
    
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

def init_model_params(
    input_dim: int, 
    hidden_sizes: list[int], 
    output_dim: int, 
    rng: np.random.Generator
) -> dict[str, Any]:
    '''
    Initialize model parameters for MLP
    Input:
        input_dim: dimension of input vector
        hidden_sizes: list of hidden layer sizes
        output_dim: number of actions
        rng: random number generator
    '''
    layer_sizes = [input_dim] + hidden_sizes + [output_dim]
    
    Ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.01
        b = np.zeros((out_dim,), dtype=np.float32)
        
        Ws.append(W)
        bs.append(b)
    
    return {"W": Ws, "b": bs}


def _apply_activation(x: np.ndarray, mode: ActivationMode) -> np.ndarray:
    if mode == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif mode == "ReLU":
        return np.maximum(x, 0.0)
    else:
        raise ValueError(f"Unsupported activation: {mode}")


def forward_logits(
    params: dict[str, list[np.ndarray]],
    x: np.ndarray,
    activation: ActivationMode,
) -> np.ndarray:
    h = x.astype(np.float32)
    Ws = params["W"]
    bs = params["b"]

    num_layers = len(Ws)
    assert num_layers == len(bs), "W/b layer count mismatch"

    for i in range(num_layers):
        W = Ws[i]
        b = bs[i]
        h = h @ W + b

        if i < num_layers - 1:
            h = _apply_activation(h, activation)

    return h


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
