from env import Game2048Env, Game2048EnvConfig
from MLP import MLPConfig
from reinforce_agent import ReinforceAgent

import logging
import sys

def log_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("game2048.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
    )


import time
from functools import wraps
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 运行耗时: {end - start:.6f} 秒")
        return result
    return wrapper

@timer
def main():
    log_setup()
    
    train_seeds = range(1)

    env_config = Game2048EnvConfig(
        obs_mode="raw",
        reward_mode="sum",
        use_action_mask=True,
    )
    env = Game2048Env(env_config)

    mlp_config = MLPConfig(
    use_onehot = False,
    num_layers = 0,
    activation = "Sigmoid"
    )

    agent = ReinforceAgent(env, mlp_config)

    policy_base_seed = int(1e6)

    for i, env_seed in enumerate(train_seeds):
        policy_seed = policy_base_seed + env_seed

        trajectory = agent.run_episode(env_seed, policy_seed)
        total_reward = trajectory["total_reward"]

        logging.info(
        "Episode %d: env_seed=%s, policy_seed=%s, total_reward=%.2f",
        i, env_seed, policy_seed, total_reward
        )
        logging.info("ENDGAME STATE\n"+env.render(mode="ansi"))

    # TODO: training logic




if __name__ == "__main__":
    main()



# 20000 without training 623s



