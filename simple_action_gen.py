import numpy as np


# ==========================
# Baseline action generators
# ==========================
def action_gen(obs, action_mask: np.ndarray) -> int:
    """
    Random valid action baseline:
    avg_reward=1103.61, max_reward=3148.00, min_reward=128.00
    """
    valid_actions = [i for i, v in enumerate(action_mask.tolist()) if v]
    return int(np.random.choice(valid_actions))


def action_gen_1(obs, action_mask: np.ndarray) -> int:
    """
    Fixed priority: Up -> Right -> Down -> Left
    avg_reward=2266.07, max_reward=6788.00, min_reward=180.00
    """
    return [i for i, v in enumerate(action_mask.tolist()) if v][0]


def action_gen_2(obs, action_mask: np.ndarray) -> int:
    """
    Fixed priority: Up -> Right -> Left -> Down
    avg_reward=2595.54, max_reward=7752.00, min_reward=448.00
    """
    for a in [0, 1, 3, 2]:
        if action_mask[a]:
            return a
    # This should theoretically never be reached (there is always at least one valid action)
    return 0