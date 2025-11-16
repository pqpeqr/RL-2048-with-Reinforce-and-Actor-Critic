

import sys
import numpy as np

from typing import Generator

from env import Game2048Env, Game2048EnvConfig
from MLP import MLPConfig
from reinforce_agent import ReinforceAgent, ReinforceAgentConfig

from timer import timer

import logging
import logging_ext

logger = logging.getLogger(__name__)


def log_setup():
    logging.basicConfig(
        level=logging.VERBOSE,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("game2048.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
    )


def make_fixed_seed_iter(base_seed: int) -> Generator[int, None, None]:
    rng = np.random.default_rng(base_seed)
    INT_64_MAX = np.iinfo(np.int64).max
    while True:
        yield int(
            rng.integers(
                low=0,
                high=INT_64_MAX,
                dtype=np.int64,
            )
        )


@timer
def evaluation():
    
    num_episodes = 50

    env_config = Game2048EnvConfig(
        obs_mode="log2",
        reward_mode="log2",
        use_action_mask=True,
    )
    env = Game2048Env(env_config)

    mlp_config = MLPConfig(
        use_onehot=False,
        num_layers=0,
        activation="Sigmoid",
    )

    agent_config = ReinforceAgentConfig(
        gamma=1,
        learning_rate=1e-3,
        baseline_mode="batch",
        model_seed=123,
    )

    agent = ReinforceAgent(env, mlp_config, agent_config)

    env_seed_iter = make_fixed_seed_iter(base_seed=3)
    policy_seed_iter = make_fixed_seed_iter(base_seed=7)

    all_rewards = []

    for ep_idx in range(num_episodes):
        logger.verbose(f"Starting evaluation episode {ep_idx+1}/{num_episodes}")
        
        env_seed = next(env_seed_iter)
        policy_seed = next(policy_seed_iter)

        trajectory = agent.run_episode(env_seed, policy_seed)
        total_reward = trajectory["total_reward"]
        all_rewards.append(total_reward)

    all_rewards = np.array(all_rewards, dtype=np.float32)
    avg_r = float(all_rewards.mean())
    max_r = float(all_rewards.max())
    min_r = float(all_rewards.min())

    logger.info(
        "Evaluation summary: episodes=%d, avg_reward=%.2f, max_reward=%.2f, min_reward=%.2f",
        len(all_rewards),
        avg_r,
        max_r,
        min_r,
    )


@timer
def training():
    
    # configs

    # 32*625 = 20_000 total
    batch_size = 32
    num_batches = 1

    env_config = Game2048EnvConfig(
        obs_mode="log2",
        reward_mode="sum",
        use_action_mask=False,
    )
    env = Game2048Env(env_config)

    mlp_config = MLPConfig(
        use_onehot=False,
        num_layers=0,
        activation="Sigmoid",
    )

    agent_config = ReinforceAgentConfig(
        gamma=0.99,
        learning_rate=1e-3,
        baseline_mode="batch",
        model_seed=0,
    )

    agent = ReinforceAgent(env, mlp_config, agent_config)

    env_seed_iter = make_fixed_seed_iter(base_seed=12345)
    policy_seed_iter = make_fixed_seed_iter(base_seed=54321)

    # running process
    for batch_idx in range(num_batches):
        batch_trajectories = []
        batch_rewards = []

        for i in range(batch_size):
            env_seed = next(env_seed_iter)
            policy_seed = next(policy_seed_iter)

            trajectory = agent.run_episode(env_seed, policy_seed)
            batch_trajectories.append(trajectory)
            total_reward = trajectory["total_reward"]
            batch_rewards.append(total_reward)
            
            logger.debug(
                "Batch: %d, Episode %d: env_seed=%s, policy_seed=%s, total_reward=%.2f",
                batch_idx, i, env_seed, policy_seed, total_reward
            )

        agent.update_batch(batch_trajectories)

        batch_rewards = np.array(batch_rewards, dtype=np.float32)
        avg_r = float(batch_rewards.mean())
        max_r = float(batch_rewards.max())
        min_r = float(batch_rewards.min())

        msg = (
            f"Batch {batch_idx + 1}/{num_batches}: "
            f"avg_reward={avg_r:.2f}, "
            f"max_reward={max_r:.2f}, "
            f"min_reward={min_r:.2f}, "
        )
        logger.info(msg)
    logger.info("Training finished.")




if __name__ == "__main__":
    log_setup()
    
    # evaluation()
    training()
    
    




