"""
2048 REINFORCE runner

Usage:
    python runner.py -conf path/to/config.json

Example structure of the configuration file (JSON):

{
  "env": {
    "size": 4,
    "obs_mode": "onehot",
    "obs_log2_scale": 1.0,
    "reward_mode": "log2",
    "base_reward_scale": 1.0,
    "bonus_mode": "off",
    "bonus_scale": 1.0,
    "step_reward": 0.0,
    "endgame_penalty": 0.0,
    "use_action_mask": true,
    "invalid_action_penalty": -1.0,
    "max_steps": null,
    "empty_tile_reward": 0.05,
    "merge_reward": 0.0
  },
  "mlp": {
    "hidden_sizes": [256, 128, 64],
    "activation": "ReLU",
    "init_distribution": "HeNormal",
    "last_init_normal": true
  },
  "agent": {
    "gamma": 0.99,
    "learning_rate": 0.01,
    "baseline_mode": "batch",
    "model_seed": 0,
    "reward_rank_weights": null,
    "optimizer": "adam",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "augmentation": false,
    "use_critic": true,
    "critic_learning_rate": 0.0005
  },
  "train": {
    "batch_size": 256,
    "num_batches": 0,
    "env_base_seed": 3,
    "policy_base_seed": 7
  },
  "eval": {
    "num_episodes": 50,
    "env_base_seed": 3,
    "policy_base_seed": 7,
    "model_path": null
  },
  "run_mode": "Training",
  "log_level": "VERBOSE"
}

Field descriptions (core parts):

- env:          passed to Game2048EnvConfig(**env)
- mlp:          passed to MLPConfig(**mlp)
- agent:        passed to ReinforceAgentConfig(**agent)
- train:        training-related parameters used by the training loop
- eval:         evaluation-related parameters
    - model_path: if not null and run_mode is "Evaluation",
                  agent.load_model(model_path) will be called to load model weights.
- run_mode:     "Training" or "Evaluation" (case-insensitive)
- log_level:    log level string such as "INFO", "DEBUG", "VERBOSE", etc.

Note:
- If -conf is not provided, DEFAULT_* configs in this file will be used,
  and the program will run in Evaluation mode by default.
"""

import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Generator, List, Dict, Any

import numpy as np

from src.env import Game2048Env, Game2048EnvConfig
from src.MLP import MLPConfig
from src.reinforce_agent import ReinforceAgent, ReinforceAgentConfig

from src.utils.timer import timer

import src.utils.logging_ext as logging_ext          # register custom log levels (e.g. VERBOSE)


logger = logging.getLogger(__name__)

# =====================================
# Global constants / default configs
# =====================================

TRAINING_HISTORY_DIR = Path("training_history")

# Default run mode (can be overridden by "run_mode" in the config file)
RUN_MODE: str = "Evaluation"

# Default environment configuration
DEFAULT_ENV_KWARGS: Dict[str, Any] = {
    "size": 4,                      # int, board size
    "obs_mode": "log2",             # str, observation encoding mode: ["raw", "log2", "onehot"]
    "obs_log2_scale": 0.0625,       # float, scaling factor when obs_mode="log2"
    "reward_mode": "log2",          # str, reward calculation mode: ["sum", "log2"]
    "base_reward_scale": 0.5,       # float, base reward scaling
    "bonus_mode": "off",            # str, extra reward mode: ["off", "raw", "log2"]
    "bonus_scale": 1.0,             # float, extra reward scaling
    "step_reward": 0.0,             # float, constant reward per step
    "endgame_penalty": 0.0,         # float, terminal penalty
    "use_action_mask": True,        # bool, whether to enable action mask
    "invalid_action_penalty": -1.0, # float, penalty for invalid actions when use_action_mask=False
    "max_steps": 1024,              # int, maximum steps per episode, None means no limit
    "empty_tile_reward": 0.0,       # float, reward for number of empty tiles
    "merge_reward": 0.0,            # float, constant reward for each merge
}

# Default network configuration
DEFAULT_MLP_KWARGS: Dict[str, Any] = {
    "hidden_sizes": [256,256],          # list[int], hidden layer sizes, empty means linear model
    "activation": "ReLU",               # str, activation function: ["Sigmoid", "ReLU"]
    "init_distribution": "HeNormal",    # str, weight init distribution: ["XavierNormal", "HeNormal", "XavierUniform", "Normal"]
    "last_init_normal": True,           # bool, whether to use standard normal init for the last layer
}

# Default agent configuration
DEFAULT_AGENT_KWARGS: Dict[str, Any] = {
    "gamma": 0.99,                  # float, discount factor in [0, 1]
    "learning_rate": 1e-4,          # float, learning rate
    "baseline_mode": "batch",       # str, baseline mode: ["off", "each", "batch", "batch_norm"]
    "model_seed": 0,                # int, random seed for model parameters
    "reward_rank_weights": None,    # list[float] | None, CVaR weight list applied to reward quantiles
    "optimizer": "sgd",             # str, optimizer type: ["sgd", "adam"]
    "adam_beta1": 0.9,              # float, Adam beta1
    "adam_beta2": 0.999,            # float, Adam beta2
    "augmentation": False,          # bool, whether to use data augmentation
    "use_critic": False,            # bool, whether to enable Critic (Actor-Critic)
    "critic_learning_rate": 1e-5,  # float, Critic learning rate
}

# Default training configuration
DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 256,          # int, number of episodes per batch
    "num_batches": 256,         # int, number of batches
    "env_base_seed": 3,         # int, base random seed for environments
    "policy_base_seed": 7,      # int, base random seed for policy
}

# Default evaluation configuration
DEFAULT_EVAL_CONFIG: Dict[str, Any] = {
    "num_episodes": 2048,       # int, number of evaluation episodes
    "env_base_seed": 12345,     # int, base random seed for evaluation environments
    "policy_base_seed": 54321,  # int, base random seed for evaluation policies
    "model_path": None,         # None | str, if not None, evaluation will attempt to load this model
}

# Default log level
DEFAULT_LOG_LEVEL_NAME = "VERBOSE"  # str, log level name ["DEBUG", "VERBOSE", "INFO"]


# ==========================
# Logging and utilities
# ==========================
def log_setup() -> None:
    """
    Initialize the base logging system:
    - Console output (INFO)

    File logs for each training/evaluation run are attached later
    via attach_run_file_logger, and are stored under training_history.
    Uses DEFAULT_LOG_LEVEL_NAME as the global log level.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    handlers: List[logging.Handler] = [stdout_handler]

    level = getattr(logging, DEFAULT_LOG_LEVEL_NAME, logging.INFO)
    fmt = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
    )



def attach_run_file_logger(log_path: Path) -> None:
    """
    Attach an additional file logger for a specific training/evaluation run.

    This removes any existing FileHandler on the root logger and adds a new one
    writing to the given log_path.
    """
    root = logging.getLogger()

    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create directory for log file: %s", log_path.parent)

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        logger.exception("Failed to create log file: %s", log_path)
        return

    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def make_fixed_seed_iter(base_seed: int) -> Generator[int, None, None]:
    """
    Generate an infinite sequence of integers that can be used as seeds
    for environments/policies.

    The sequence is deterministic given the same base_seed, which enables
    reproducible experiments.
    """
    rng = np.random.default_rng(base_seed)
    int_64_max = np.iinfo(np.int64).max
    while True:
        yield int(
            rng.integers(
                low=0,
                high=int_64_max,
                dtype=np.int64,
            )
        )


def build_full_config_dict(
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Pack env/mlp/agent config objects and extra info into one dict,
    convenient for saving as JSON.
    """
    import dataclasses

    def to_dict(obj: Any) -> Any:
        try:
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
        except Exception:
            pass

        if hasattr(obj, "__dict__"):
            return vars(obj)
        return obj

    cfg = {
        "env": to_dict(env_config),
        "mlp": to_dict(mlp_config),
        "agent": to_dict(agent_config),
    }
    if extra:
        cfg.update(extra)
    return cfg


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Safely write a dict to a JSON file.

    If writing fails, it only logs an error without terminating the program.
    """
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")
    except Exception:
        logger.exception("Failed to write JSON config to %s", path)


def safe_append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """
    Append a row to a CSV file.
    If the file does not exist, write the header first.
    """
    try:
        file_exists = path.exists() and path.stat().st_size > 0
    except OSError:
        file_exists = False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create directory for CSV: %s", path.parent)

    try:
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        logger.exception("Failed to append row to CSV %s", path)


def create_training_run_dir() -> tuple[Path, str]:
    """
    Create a directory for a single training run:
    - training_history/<timestamp>/

    Returns (run_dir, run_id_str).
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = TRAINING_HISTORY_DIR

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception(
            "Failed to create training_history directory at %s, fallback to current directory.",
            base_dir,
        )
        base_dir = Path.cwd()

    run_dir = base_dir / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception(
            "Failed to create run directory %s, fallback to base dir %s.",
            run_dir,
            base_dir,
        )
        run_dir = base_dir

    return run_dir, run_id


def create_evaluation_log_path() -> tuple[Path, str]:
    """
    Create a log file path for a single evaluation:
    - training_history/eval_<timestamp>.log

    Returns (log_path, eval_id_str).
    """
    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = TRAINING_HISTORY_DIR

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception(
            "Failed to create training_history directory at %s, fallback to current directory.",
            base_dir,
        )
        base_dir = Path.cwd()

    log_path = base_dir / f"eval_{eval_id}.log"
    return log_path, eval_id


# ==========================
# Config file loading logic
# ==========================
def apply_config_overrides_from_dict(conf: Dict[str, Any]) -> None:
    """
    Override DEFAULT_* global configs and RUN_MODE / DEFAULT_LOG_LEVEL_NAME
    based on the given config dict.

    Expected structure:
        {
          "env": {...},
          "mlp": {...},
          "agent": {...},
          "train": {...},
          "eval": {...},
          "run_mode": "Training" | "Evaluation",
          "log_level": "VERBOSE" | "INFO" | ...
        }
    """
    global DEFAULT_ENV_KWARGS, DEFAULT_MLP_KWARGS, DEFAULT_AGENT_KWARGS
    global DEFAULT_TRAIN_CONFIG, DEFAULT_EVAL_CONFIG
    global RUN_MODE, DEFAULT_LOG_LEVEL_NAME

    env_cfg = conf.get("env")
    if isinstance(env_cfg, dict):
        DEFAULT_ENV_KWARGS.update(env_cfg)

    mlp_cfg = conf.get("mlp")
    if isinstance(mlp_cfg, dict):
        DEFAULT_MLP_KWARGS.update(mlp_cfg)

    agent_cfg = conf.get("agent")
    if isinstance(agent_cfg, dict):
        DEFAULT_AGENT_KWARGS.update(agent_cfg)

    train_cfg = conf.get("train")
    if isinstance(train_cfg, dict):
        DEFAULT_TRAIN_CONFIG.update(train_cfg)

    eval_cfg = conf.get("eval")
    if isinstance(eval_cfg, dict):
        DEFAULT_EVAL_CONFIG.update(eval_cfg)

    run_mode = conf.get("run_mode")
    if isinstance(run_mode, str):
        # Normalize case so users can write "training", "TRAINING", etc.
        RUN_MODE = run_mode

    log_level = conf.get("log_level")
    if isinstance(log_level, str):
        DEFAULT_LOG_LEVEL_NAME = log_level


def load_config_from_file(path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from a JSON file and apply it to
    DEFAULT_* / RUN_MODE / DEFAULT_LOG_LEVEL_NAME.

    If reading or parsing fails, print an error and exit.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
        conf = json.loads(text)
    except Exception as e:
        print(f"[runner] Failed to load config file '{p}': {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(conf, dict):
        print(f"[runner] Config file '{p}' must contain a JSON object at root.", file=sys.stderr)
        sys.exit(1)

    apply_config_overrides_from_dict(conf)
    return conf


# ==========================
# Training logic
# ==========================
def build_training_components() -> tuple[
    Game2048Env,
    ReinforceAgent,
    Game2048EnvConfig,
    MLPConfig,
    ReinforceAgentConfig,
    Dict[str, Any],
]:
    """
    Build environment, network, and agent for training based on DEFAULT_* configs.
    These DEFAULT_* can be overridden by the configuration file.
    """
    env_config = Game2048EnvConfig(**DEFAULT_ENV_KWARGS)
    env = Game2048Env(env_config)

    mlp_config = MLPConfig(**DEFAULT_MLP_KWARGS)
    agent_config = ReinforceAgentConfig(**DEFAULT_AGENT_KWARGS)

    agent = ReinforceAgent(env, mlp_config, agent_config)

    train_extra_cfg = dict(DEFAULT_TRAIN_CONFIG)

    return env, agent, env_config, mlp_config, agent_config, train_extra_cfg


def training_loop(
    env: Game2048Env,
    agent: ReinforceAgent,
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
    train_cfg: Dict[str, Any],
    run_dir: Path,
    run_id: str,
) -> None:
    """
    Training loop:
    - Fixed batch_size / num_batches / seeds
    - For each batch:
        * collect trajectories
        * compute statistics
        * update model
    - Writes:
        * log (train_xxx.log)
        * config.json
        * training_stats.csv (append per batch)
        * model checkpoints (when avg reward improves and step > 30)

    If num_batches == 0, the loop runs indefinitely until interrupted.
    """
    batch_size = int(train_cfg["batch_size"])
    num_batches = int(train_cfg["num_batches"])
    env_base_seed = int(train_cfg["env_base_seed"])
    policy_base_seed = int(train_cfg["policy_base_seed"])

    # Switch to log file for this training run
    log_file_path = run_dir / f"train_{run_id}.log"
    attach_run_file_logger(log_file_path)
    logger.info("Training run directory: %s", run_dir)

    # Save config snapshot
    full_cfg = build_full_config_dict(
        env_config=env_config,
        mlp_config=mlp_config,
        agent_config=agent_config,
        extra={
            "run_mode": "Training",
            "train": train_cfg,
        },
    )
    config_path = run_dir / "config.json"
    safe_write_json(config_path, full_cfg)

    logger.info(
        "Training full config:\n%s",
        json.dumps(full_cfg, ensure_ascii=False, indent=2),
    )

    # Prepare CSV
    csv_path = run_dir / "training_stats.csv"
    csv_fieldnames = ["batch", "avg_reward", "max_reward", "min_reward", "max_tile_counts"]

    # Seed generators
    env_seed_iter = make_fixed_seed_iter(base_seed=env_base_seed)
    policy_seed_iter = make_fixed_seed_iter(base_seed=policy_base_seed)

    # Max-tile values to track
    tile_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    tile_index = {t: i for i, t in enumerate(tile_values)}

    best_avg_reward = float("-inf")

    if num_batches > 0:
        logger.info(
            "Start training: batch_size=%d, num_batches=%d, env_base_seed=%d, policy_base_seed=%d",
            batch_size,
            num_batches,
            env_base_seed,
            policy_base_seed,
        )
    else:
        logger.info(
            "Start training: batch_size=%d, num_batches=infinite, env_base_seed=%d, policy_base_seed=%d",
            batch_size,
            env_base_seed,
            policy_base_seed,
        )

    global_step = 0  # number of batches completed

    # num_batches == 0 means infinite loop
    while True:
        batch_trajectories = []
        batch_rewards: List[float] = []
        batch_max_tiles: List[int] = []

        # Collect one batch of episodes
        for i in range(batch_size):
            env_seed = next(env_seed_iter)
            policy_seed = next(policy_seed_iter)

            trajectory = agent.run_episode(env_seed, policy_seed)
            batch_trajectories.append(trajectory)

            total_reward = float(trajectory["total_reward"])
            batch_rewards.append(total_reward)

            max_tile = int(trajectory.get("max_tile", 0))
            batch_max_tiles.append(max_tile)

            logger.debug(
                "Batch: %d, Episode %d: env_seed=%s, policy_seed=%s, total_reward=%.2f, max_tile=%s",
                global_step + 1,  # More intuitive, start from 1
                i,
                env_seed,
                policy_seed,
                total_reward,
                max_tile,
            )

        global_step += 1

        # Reward statistics
        batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
        avg_r = float(batch_rewards_np.mean())
        max_r = float(batch_rewards_np.max())
        min_r = float(batch_rewards_np.min())

        # Max-tile statistics
        counts_for_batch = [0] * len(tile_values)
        for mt in batch_max_tiles:
            idx = tile_index.get(mt)
            if idx is not None:
                counts_for_batch[idx] += 1

        max_tile_counts_dict = {t: c for t, c in zip(tile_values, counts_for_batch)}

        # Log with max-tile statistics
        if num_batches > 0:
            batch_info = f"{global_step}/{num_batches}"
        else:
            batch_info = f"{global_step} (infinite)"

        msg = (
            f"Batch {batch_info}: "
            f"avg_reward={avg_r:.2f}, "
            f"max_reward={max_r:.2f}, "
            f"min_reward={min_r:.2f}, "
            f"max_tile_counts={max_tile_counts_dict}"
        )
        logger.info(msg)

        # Checkpoint saving logic (before update)
        is_record = avg_r > best_avg_reward
        if is_record:
            best_avg_reward = avg_r

        if global_step > 30 and is_record:
            time_str_short = datetime.now().strftime("%H%M%S")
            model_file_name = f"model_{time_str_short}_step{global_step}.npz"
            model_path = run_dir / model_file_name
            try:
                agent.save_model(str(model_path))
                logger.info(
                    "Saved model checkpoint: step=%d, avg_reward=%.4f, file=%s",
                    global_step,
                    avg_r,
                    model_path,
                )
            except Exception:
                logger.exception("Failed to save model checkpoint to %s", model_path)

        # Update policy using current batch
        agent.update_batch(batch_trajectories)

        # Write CSV row
        row = {
            "batch": global_step,
            "avg_reward": avg_r,
            "max_reward": max_r,
            "min_reward": min_r,
            "max_tile_counts": json.dumps(counts_for_batch, ensure_ascii=False),
        }
        safe_append_csv_row(csv_path, csv_fieldnames, row)

        # Stop if we have a finite number of batches and reached the limit
        if num_batches > 0 and global_step >= num_batches:
            break

    logger.info("Training finished.")


@timer
def training() -> None:
    """
    Training entry point.

    Uses current DEFAULT_* configs to perform a full training run and
    archives it under training_history.
    DEFAULT_* can be overridden by the configuration file.
    """
    env, agent, env_config, mlp_config, agent_config, train_cfg = build_training_components()
    run_dir, run_id = create_training_run_dir()
    training_loop(env, agent, env_config, mlp_config, agent_config, train_cfg, run_dir, run_id)


# ==========================
# Evaluation logic
# ==========================
def build_evaluation_components() -> tuple[
    Game2048Env,
    ReinforceAgent,
    Game2048EnvConfig,
    MLPConfig,
    ReinforceAgentConfig,
    Dict[str, Any],
]:
    """
    Build environment, network, and agent for evaluation based on DEFAULT_* configs.

    If DEFAULT_EVAL_CONFIG["model_path"] is not None, it tries to load
    that model's weights.
    """
    env_config = Game2048EnvConfig(**DEFAULT_ENV_KWARGS)
    env = Game2048Env(env_config)

    mlp_config = MLPConfig(**DEFAULT_MLP_KWARGS)
    agent_config = ReinforceAgentConfig(**DEFAULT_AGENT_KWARGS)

    agent = ReinforceAgent(env, mlp_config, agent_config)

    eval_extra_cfg = dict(DEFAULT_EVAL_CONFIG)

    model_path = eval_extra_cfg.get("model_path")
    if model_path:
        try:
            agent.load_model(str(model_path))
            logger.info("Loaded model from '%s' for evaluation.", model_path)
        except Exception:
            logger.exception(
                "Failed to load model from '%s', continue with randomly initialized weights.",
                model_path,
            )

    return env, agent, env_config, mlp_config, agent_config, eval_extra_cfg


def evaluation_loop(
    env: Game2048Env,
    agent: ReinforceAgent,
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
    eval_cfg: Dict[str, Any],
) -> None:
    """
    Evaluation loop:
    - Use fixed number of episodes and seeds
    - Run agent.run_episode(..., use_greedy=True) for policy execution
    - Collect statistics of rewards and max_tile distribution
    - Log overall summary and the state sequence of the highest-scoring episode
    """
    num_episodes = int(eval_cfg["num_episodes"])
    env_base_seed = int(eval_cfg["env_base_seed"])
    policy_base_seed = int(eval_cfg["policy_base_seed"])

    env_seed_iter = make_fixed_seed_iter(base_seed=env_base_seed)
    policy_seed_iter = make_fixed_seed_iter(base_seed=policy_base_seed)

    all_rewards: List[float] = []
    max_rewards = 0.0
    max_states: List[Any] = []

    maxtile_counts: Dict[int, int] = defaultdict(int)

    full_cfg = build_full_config_dict(
        env_config=env_config,
        mlp_config=mlp_config,
        agent_config=agent_config,
        extra={
            "run_mode": "Evaluation",
            "eval": eval_cfg,
        },
    )
    logger.info(
        "Evaluation full config:\n%s",
        json.dumps(full_cfg, ensure_ascii=False, indent=2),
    )

    logger.info(
        "Start evaluation: episodes=%d, env_base_seed=%d, policy_base_seed=%d",
        num_episodes,
        env_base_seed,
        policy_base_seed,
    )

    for ep_idx in range(num_episodes):
        logger.verbose("Starting evaluation episode %d/%d", ep_idx + 1, num_episodes)

        env_seed = next(env_seed_iter)
        policy_seed = next(policy_seed_iter)

        trajectory = agent.run_episode(env_seed, policy_seed, action_gen=None, use_greedy=True)
        total_reward = float(trajectory["total_reward"])
        all_rewards.append(total_reward)

        if total_reward > max_rewards:
            max_states = trajectory.get("states", [])
            max_rewards = total_reward

        maxtile_counts[int(trajectory.get("max_tile", 0))] += 1

    all_rewards_np = np.array(all_rewards, dtype=np.float32)
    avg_r = float(all_rewards_np.mean())
    max_r = float(all_rewards_np.max())
    min_r = float(all_rewards_np.min())

    logger.info(
        "Evaluation summary: episodes=%d, avg_reward=%.2f, max_reward=%.2f, min_reward=%.2f",
        len(all_rewards),
        avg_r,
        max_r,
        min_r,
    )

    max_tile_summary = {
        tile: {
            "count": count,
            "pct": round(count / num_episodes * 100.0, 2),
        }
        for tile, count in sorted(maxtile_counts.items())
    }
    logger.info("Max tile counts: %s", max_tile_summary)

    # Print the state sequence for the highest-scoring episode
    for state in max_states:
        logger.verbose("\n%s\n%s", state, "-" * 20)


@timer
def evaluation() -> None:
    """
    Evaluation entry point.

    Uses current DEFAULT_* configs to perform a full evaluation and
    write logs to training_history/eval_*.log.

    If eval.model_path is provided in the config, the model is loaded
    before evaluation.
    """
    env, agent, env_config, mlp_config, agent_config, eval_cfg = build_evaluation_components()

    eval_log_path, eval_id = create_evaluation_log_path()
    attach_run_file_logger(eval_log_path)
    logger.info("Evaluation log file: %s (id=%s)", eval_log_path, eval_id)

    evaluation_loop(env, agent, env_config, mlp_config, agent_config, eval_cfg)


# ==========================
# Command-line entry
# ==========================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Only keep the configuration file argument:
      -conf / --conf: path to JSON configuration file

    All other detailed parameters are managed via the config file.
    """
    parser = argparse.ArgumentParser(
        description="2048 REINFORCE runner (configuration via JSON file)."
    )

    parser.add_argument(
        "-conf",
        "--conf",
        dest="conf",
        type=str,
        help="Path to configuration JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If a config file is provided, load it and override DEFAULT_* / RUN_MODE / DEFAULT_LOG_LEVEL_NAME
    if args.conf:
        load_config_from_file(args.conf)

    # Initialize logging (this uses DEFAULT_LOG_LEVEL_NAME, which may have been overridden)
    log_setup()
    logger.info("Runner started with mode=%s", RUN_MODE)

    # Select training or evaluation based on RUN_MODE (case-insensitive)
    if RUN_MODE.lower() == "training":
        training()
    else:
        evaluation()
