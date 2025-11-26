# app.py

"""
Streamlit visualization console: training and evaluation dashboard for 2048 environment + REINFORCE algorithm.

Current features:

1. Interactive configuration:
   - Environment (Game2048EnvConfig)
   - Policy network (MLPConfig)
   - Agent / algorithm (ReinforceAgentConfig)

2. Two run modes:
   - Training:
       * Online training for REINFORCE agent
       * Dynamic batch-level reward curves (avg / max / min)
       * Dynamic line chart for counts of each max_tile across batches
   - Evaluation:
       * Evaluate current policy over multiple episodes
       * Plot per-episode reward curve
       * Collect max_tile distribution (table + pie chart)
       * Configurable greedy / stochastic evaluation (use_greedy)

3. Configuration persistence:
   - Auto load / save config.json
   - On startup, restore all UI parameters from config.json (including training / evaluation defaults and run mode)

4. Training / evaluation logging and result archiving:
   - Root log outputs to stdout and app.log
   - Every Training run:
       * Create a unique directory under ./training_history/<timestamp>/
       * In that directory, save:
           - The config.json snapshot for this run (full env/mlp/agent/train config)
           - Training log train_*.log
           - Training statistics CSV: training_stats.csv (per-batch reward statistics + max_tile statistics)
           - When batch average reward exceeds the historical best AND global_step > 30,
             save a model snapshot model_*.npz
   - Every Evaluation:
       * Create an eval_*.log file in ./training_history/
       * Optionally load model parameters from a specified file before evaluation (eval.model_path)
"""

import sys
import logging
from typing import Generator, List
import json
from pathlib import Path
import dataclasses
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.env import Game2048Env, Game2048EnvConfig
from src.MLP import MLPConfig
from src.reinforce_agent import ReinforceAgent, ReinforceAgentConfig

import src.utils.logging_ext as logging_ext     # add VERBOSE level to logging


# ==========================
# Matplotlib global font config
# ==========================
# Support Chinese fonts (SimHei / Microsoft YaHei) and correct minus sign display.
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================
# Logging initialization
# ==========================
logger = logging.getLogger(__name__)  # module-level logger


def _get_log_level(level_name: str) -> int:
    """
    Map string log level to logging module integer level.
    """
    level_name = (level_name or "INFO").upper()
    if level_name == "DEBUG":
        return logging.DEBUG
    elif level_name == "VERBOSE":
        return logging.VERBOSE
    else:
        return logging.INFO


def log_setup(level_name: str = "INFO") -> None:
    """
    Initialize the logging system (only effective on first call):

    - Use level_name ("DEBUG" / "VERBOSE" / "INFO") to set logging.basicConfig level.
    - If the root logger already has handlers, only update its level and do not reconfigure handlers.
    - Default configuration:
        * Console output (stdout, INFO level)
        * File output (app.log, DEBUG level)
    """
    root = logging.getLogger()
    level = _get_log_level(level_name)

    if root.handlers:
        # Already initialized, only update the root level.
        root.setLevel(level)
        return

    # Console output (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    # File output (default app.log, later redirected into training_history per run)
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    fmt = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[file_handler, stdout_handler],
    )


# ==========================
# Common config packing helper
# ==========================
def build_full_config_dict(
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
    extra: dict | None = None,
) -> dict:
    """
    Merge environment, network, and agent config objects plus extra fields into
    a JSON-serializable dict for logging or saving.

    Args:
        env_config: Game2048EnvConfig instance
        mlp_config: MLPConfig instance
        agent_config: ReinforceAgentConfig instance
        extra: additional config items (e.g., current run_mode, training/evaluation params)

    Returns:
        dict suitable for json.dumps
    """

    def to_dict(obj):
        """
        Try to convert an arbitrary object to dict:
        - If it is a dataclass, use dataclasses.asdict
        - Else try __dict__ / vars
        - Otherwise return it as-is (assuming primitive or nested structure)
        """
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


# ==========================
# Config file read/write (config.json)
# ==========================
CONFIG_PATH = Path("config.json")


def load_config() -> dict:
    """
    Load configuration from config.json.

    - If file does not exist, return an empty dict.
    - If file exists but cannot be parsed (corrupted / not JSON), ignore and return empty dict.
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    """
    Save configuration dict to config.json.

    - Uses UTF-8 encoding
    - ensure_ascii=False to preserve non-ASCII content
    """
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==========================
# Fixed random seed generator
# ==========================
def make_fixed_seed_iter(base_seed: int) -> Generator[int, None, None]:
    """
    Build an infinite iterator that yields int64 random numbers from np.random.default_rng.

    Usage:
    - Use base_seed to seed RNG, then generate a deterministic sequence of integer seeds
      for environment and policy, making training/evaluation reproducible under the same config.

    Args:
        base_seed: base seed for RNG

    Returns:
        generator that yields int values in int64 range
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


# ==========================
# Training mode: training + dynamic plots
# ==========================
def run_training(
    env: Game2048Env,
    agent: ReinforceAgent,
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
) -> None:
    """
    "Training Mode" part of the Streamlit page.

    Main workflow:
    1. Configure training parameters on the page:
       - batch_size, num_batches
       - environment and policy seed bases env_base_seed / policy_base_seed
    2. After pressing "Start Training":
       - For each batch:
         * Use env_seed_iter / policy_seed_iter to call agent.run_episode
           and collect batch_size trajectories;
         * Compute batch avg / max / min reward;
         * Count per-episode max_tile distribution within the batch (only tiles 16~4096).
       - During training:
         * Maintain global best_avg_reward
         * When global_step > 30 AND current batch avg_reward > best_avg_reward:
           save a model snapshot to this run directory, then call agent.update_batch;
         * Write all batch statistics into DataFrame, and append to
           <run_dir>/training_stats.csv;
         * Dynamically update two figures:
             1) Training reward curves (avg / max / min)
             2) Max tile counts per batch (per tile value)
         * Show all batch statistics as a DataFrame (latest batch at the top).
    3. All outputs of this run (logs/config/CSV/models) are archived under
       ./training_history/<run_start_time>/.
    """
    st.header("Training Mode")

    # -------- Training hyperparameters (batch_size, num_batches, approximate total episodes) --------
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input(
            "batch_size",
            min_value=1,
            max_value=1024,
            step=1,
            key="batch_size",
        )
    with col2:
        num_batches = st.number_input(
            "num_batches (0 = infinite)",
            min_value=0,
            max_value=10_000,
            step=1,
            key="num_batches",
        )
    with col3:
        if num_batches == 0:
            st.write("Approximate total episodes: **∞** (infinite)")
        else:
            max_episodes = int(batch_size * num_batches)
            st.write(f"Approximate total episodes: **{max_episodes}**")

    # -------- Seed settings --------
    st.subheader("Random Seed Settings")
    env_base_seed = st.number_input(
        "env_seed base",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="env_base_seed_train",
    )
    policy_base_seed = st.number_input(
        "policy_seed base",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="policy_base_seed_train",
    )

    # -------- Start training button --------
    start_button = st.button("Start Training")

    if not start_button:
        st.info("Click the button above to start training.")
        return

    # ======== Create dedicated directory and log file for this training run ========
    run_start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_history_dir = Path.cwd() / "training_history"
    run_dir = base_history_dir / run_start_time_str
    run_dir.mkdir(parents=True, exist_ok=True)

    # Redirect FileHandler to run_dir
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    log_file_path = run_dir / f"train_{run_start_time_str}.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    csv_stats_path = run_dir / "training_stats.csv"
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Training run directory: %s", str(run_dir))

    # -------- Log full configuration (including run_mode and training params) --------
    full_cfg = build_full_config_dict(
        env_config=env_config,
        mlp_config=mlp_config,
        agent_config=agent_config,
        extra={
            "run_mode": "Training",
            "train": {
                "batch_size": batch_size,
                "num_batches": num_batches,
                "env_base_seed": env_base_seed,
                "policy_base_seed": policy_base_seed,
            },
        },
    )
    logger.info(
        "Training full config:\n%s",
        json.dumps(full_cfg, ensure_ascii=False, indent=2),
    )

    # Save this run's config snapshot
    run_config_path = run_dir / "config.json"
    run_config_path.write_text(
        json.dumps(full_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Start training: batch_size=%d, num_batches=%d", batch_size, num_batches)

    # Build deterministic seed iterators
    env_seed_iter = make_fixed_seed_iter(base_seed=env_base_seed)
    policy_seed_iter = make_fixed_seed_iter(base_seed=policy_base_seed)

    # -------- Per-batch reward statistics --------
    batch_avg_rewards: List[float] = []
    batch_max_rewards: List[float] = []
    batch_min_rewards: List[float] = []

    # Focus on max_tiles from 16 to 4096
    tile_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    tile_index = {t: i for i, t in enumerate(tile_values)}

    # Per-batch max_tile counts: each entry is a list of length len(tile_values)
    batch_max_tile_counts: List[List[int]] = []

    # History for line chart: key = tile value, value = list of counts over batches
    max_tile_history = {t: [] for t in tile_values}

    # ======== Best mean reward tracking ========
    best_avg_reward: float = float("-inf")

    # -------- Streamlit placeholders (progress, status, plots, tables) --------
    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_placeholder = st.empty()
    table_placeholder = st.empty()
    max_tile_plot_placeholder = st.empty()

    # -------- Train by batch --------
    batch_idx = 0
    while True:
        batch_trajectories = []
        batch_rewards = []
        batch_max_tiles = []

        # --- Collect one batch of episodes ---
        for i in range(batch_size):
            env_seed = next(env_seed_iter)
            policy_seed = next(policy_seed_iter)

            trajectory = agent.run_episode(env_seed, policy_seed)
            batch_trajectories.append(trajectory)

            total_reward = trajectory["total_reward"]
            batch_rewards.append(total_reward)

            max_tile = trajectory.get("max_tile", 0)
            batch_max_tiles.append(max_tile)

            logger.debug(
                "Batch: %d, Episode %d: env_seed=%s, policy_seed=%s, total_reward=%.2f, max_tile=%s",
                batch_idx, i, env_seed, policy_seed, total_reward, max_tile
            )

        # --- Compute reward statistics for this batch (before update_batch) ---
        batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
        avg_r = float(batch_rewards_np.mean())
        max_r = float(batch_rewards_np.max())
        min_r = float(batch_rewards_np.min())

        batch_avg_rewards.append(avg_r)
        batch_max_rewards.append(max_r)
        batch_min_rewards.append(min_r)

        # --- Collect max_tile distribution for this batch (only 16~4096) ---
        counts_for_batch = [0] * len(tile_values)
        if batch_max_tiles:
            for mt in batch_max_tiles:
                idx = tile_index.get(mt, None)
                if idx is not None:
                    counts_for_batch[idx] += 1

        batch_max_tile_counts.append(counts_for_batch)

        for j, t in enumerate(tile_values):
            max_tile_history[t].append(counts_for_batch[j])

        # --- Logging + UI status ---
        if num_batches == 0:
            msg = (
                f"Batch {batch_idx + 1} (infinite): "
                f"avg_reward={avg_r:.2f}, "
                f"max_reward={max_r:.2f}, "
                f"min_reward={min_r:.2f}"
            )
        else:
            msg = (
                f"Batch {batch_idx + 1}/{num_batches}: "
                f"avg_reward={avg_r:.2f}, "
                f"max_reward={max_r:.2f}, "
                f"min_reward={min_r:.2f}"
            )
        logger.info(msg)
        status_text.text(msg)

        # ======== Model saving logic based on mean reward record ========
        global_step = batch_idx + 1  # number of batches
        is_record = False
        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            is_record = True

        # When global_step > 30 and current avg_r beats the best so far, save model
        # before performing the policy update.
        if global_step > 30 and is_record:
            time_str_short = datetime.now().strftime("%H%M%S")
            model_file_name = f"model_{time_str_short}_step{global_step}.npz"
            model_file_path = run_dir / model_file_name

            agent.save_model(str(model_file_path))

            logger.info(
                "Saved model checkpoint: step=%d, avg_reward=%.4f, file=%s",
                global_step,
                avg_r,
                model_file_path,
            )

        # --- Perform one policy update using this batch (after saving) ---
        agent.update_batch(batch_trajectories)

        # --- Update progress bar ---
        if num_batches > 0:
            progress = int(global_step / num_batches * 100)
            progress_bar.progress(progress)
        else:
            progress_bar.progress(0)

        # --- Reward curve plot (dynamic) ---
        fig, ax = plt.subplots()
        x = np.arange(1, len(batch_avg_rewards) + 1)
        ax.plot(x, batch_avg_rewards, label="avg_reward")
        ax.plot(x, batch_max_rewards, label="max_reward")
        ax.plot(x, batch_min_rewards, label="min_reward")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward Curves")
        ax.legend()
        ax.grid(True)

        plot_placeholder.pyplot(fig)
        plt.close(fig)

        # --- DataFrame with all batch statistics ---
        df_rewards = pd.DataFrame(
            {
                "batch": x,
                "avg_reward": batch_avg_rewards,
                "max_reward": batch_max_rewards,
                "min_reward": batch_min_rewards,
                "max_tile_counts": batch_max_tile_counts,
            }
        )

        # ---- CSV: append only the last row in batch order ----
        last_row_df = df_rewards.tail(1)
        write_header = not csv_stats_path.exists()
        last_row_df.to_csv(
            csv_stats_path,
            index=False,
            encoding="utf-8",
            mode="a",
            header=write_header,
        )

        # ---- UI: show DataFrame with latest batch at the top ----
        df_ui = df_rewards.sort_values("batch", ascending=False)
        table_placeholder.dataframe(df_ui, width="stretch")

        # --- Max tile counts over batches plot ---
        fig2, ax2 = plt.subplots()
        x_line = np.arange(1, len(batch_avg_rewards) + 1)

        for t in tile_values:
            series = max_tile_history[t]
            if series and max(series) > 0:
                ax2.plot(x_line, series, label=f"{t}")

        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Episode Count")
        ax2.set_title("Max Tile Counts per Batch")
        ax2.grid(True)
        ax2.legend(title="Max Tile", fontsize="small")

        max_tile_plot_placeholder.pyplot(fig2)
        plt.close(fig2)

        # Stop when we reach the specified number of batches; when num_batches = 0, loop indefinitely.
        if num_batches > 0 and global_step >= num_batches:
            break

        batch_idx += 1

    st.success("Training finished!")


# ==========================
# Evaluation mode: multiple episodes + statistics
# ==========================
def run_evaluation(
    env: Game2048Env,
    agent: ReinforceAgent,
    env_config: Game2048EnvConfig,
    mlp_config: MLPConfig,
    agent_config: ReinforceAgentConfig,
) -> None:
    """
    "Evaluation Mode" part of the Streamlit page.

    Main workflow:
    1. Configure evaluation parameters:
       - num_episodes
       - env_base_seed / policy_base_seed
       - use_greedy (greedy vs stochastic policy)
    2. Optionally load model parameters from a specified file:
       - If model_path is set, call agent.load_model before evaluation;
       - If loading fails, show error and abort evaluation.
    3. After pressing "Start Evaluation":
       - Loop for num_episodes:
         * Use env_seed_iter / policy_seed_iter to call agent.run_episode;
         * Record each episode's total_reward and max_tile;
         * Update progress bar and status text.
       - After finishing:
         * Compute and display mean / max / min reward;
         * Plot per-episode reward curve;
         * Collect max_tile distribution (table with count/ratio);
         * Plot a pie chart of max_tile ratios.
    4. An evaluation log file eval_*.log is created in ./training_history/
       to record the evaluation configuration and statistics.
    """
    st.header("Evaluation Mode")

    # -------- Evaluation configuration: episode count, seeds, optional model file path --------
    num_episodes = st.number_input(
        "Number of evaluation episodes",
        min_value=1,
        max_value=1000,
        step=1,
        key="num_episodes",
    )
    env_base_seed = st.number_input(
        "Evaluation env_seed base",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="env_base_seed_eval",
    )
    policy_base_seed = st.number_input(
        "Evaluation policy_seed base",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="policy_base_seed_eval",
    )

    use_greedy = st.checkbox(
        "Use greedy policy during evaluation (disable to sample stochastically)",
        key="eval_use_greedy",
    )

    model_path = st.text_input(
        "Model parameters file path (optional)",
        value="",
        key="eval_model_params_path",
        help="If set, the agent will load model parameters from this file before evaluation.",
    )

    eval_button = st.button("Start Evaluation")

    if not eval_button:
        st.info(
            "Click the button to evaluate the current policy. "
            "If you want to evaluate a trained model, please specify the model parameters file path above."
        )
        return

    # ======== Create dedicated log file for this evaluation ========
    eval_start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_history_dir = Path.cwd() / "training_history"
    base_history_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    eval_log_file_path = base_history_dir / f"eval_{eval_start_time_str}.log"
    file_handler = logging.FileHandler(eval_log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Evaluation log file: %s", str(eval_log_file_path))

    # Load model parameters if a path is provided
    if model_path.strip():
        path_str = model_path.strip()
        try:
            agent.load_model(path_str)
            logger.info("Evaluation: model parameters loaded from %s", path_str)
        except Exception as e:
            st.error(f"Failed to load model parameters from '{path_str}': {e}")
            logger.exception("Failed to load model parameters for evaluation from %s", path_str)
            return

    # -------- Log configuration --------
    cfg_model_path = model_path or None

    full_cfg = build_full_config_dict(
        env_config=env_config,
        mlp_config=mlp_config,
        agent_config=agent_config,
        extra={
            "run_mode": "Evaluation",
            "eval": {
                "num_episodes": num_episodes,
                "env_base_seed": env_base_seed,
                "policy_base_seed": policy_base_seed,
                "model_path": cfg_model_path,
                "use_greedy": use_greedy,
            },
        },
    )
    logger.info(
        "Evaluation full config:\n%s",
        json.dumps(full_cfg, ensure_ascii=False, indent=2),
    )

    logger.info(
        "Start evaluation: num_episodes=%d, env_base_seed=%d, policy_base_seed=%d, use_greedy=%s",
        num_episodes,
        env_base_seed,
        policy_base_seed,
        use_greedy,
    )

    # Build seed generators
    env_seed_iter = make_fixed_seed_iter(base_seed=env_base_seed)
    policy_seed_iter = make_fixed_seed_iter(base_seed=policy_base_seed)

    all_rewards = []
    all_max_tiles = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # -------- Evaluate per episode --------
    for ep_idx in range(num_episodes):
        env_seed = next(env_seed_iter)
        policy_seed = next(policy_seed_iter)

        trajectory = agent.run_episode(env_seed, policy_seed, use_greedy=use_greedy)
        total_reward = trajectory["total_reward"]
        max_tile = trajectory.get("max_tile", 0)

        all_rewards.append(total_reward)
        all_max_tiles.append(max_tile)

        status_text.text(
            f"Episode {ep_idx + 1}/{num_episodes}, total_reward={total_reward:.2f}, max_tile={max_tile}"
        )
        progress = int((ep_idx + 1) / num_episodes * 100)
        progress_bar.progress(progress)

    # -------- Reward statistics --------
    all_rewards_np = np.array(all_rewards, dtype=np.float32)
    avg_r = float(all_rewards_np.mean())
    max_r = float(all_rewards_np.max())
    min_r = float(all_rewards_np.min())

    st.subheader("Evaluation Statistics")
    st.write(f"- Episodes: **{len(all_rewards)}**")
    st.write(f"- Average reward: **{avg_r:.2f}**")
    st.write(f"- Max reward: **{max_r:.2f}**")
    st.write(f"- Min reward: **{min_r:.2f}**")

    logger.info(
        "Evaluation finished: episodes=%d, avg_reward=%.2f, max_reward=%.2f, min_reward=%.2f",
        len(all_rewards),
        avg_r,
        max_r,
        min_r,
    )

    # -------- Reward curve (per episode) --------
    fig, ax = plt.subplots()
    x = np.arange(1, len(all_rewards_np) + 1)
    ax.plot(x, all_rewards_np)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Evaluation Episode Reward Curve")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    # -------- Max tile distribution + pie chart --------
    if all_max_tiles:
        all_max_tiles_np = np.array(all_max_tiles, dtype=np.int64)
        unique_tiles, counts = np.unique(all_max_tiles_np, return_counts=True)
        ratios = counts / float(num_episodes)

        df_max_tile = pd.DataFrame(
            {
                "max_tile": unique_tiles,
                "count": counts,
                "ratio": ratios,
            }
        ).sort_values("max_tile")

        st.subheader("Evaluation Max Tile Distribution")
        st.dataframe(df_max_tile, width="stretch")

        fig2, ax2 = plt.subplots()

        sizes = df_max_tile["ratio"].to_numpy(dtype=float)
        labels = df_max_tile["max_tile"].astype(int).astype(str).tolist()

        ax2.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Distribution of Max Tiles Over Evaluation Episodes")
        ax2.axis("equal")

        st.pyplot(fig2)
        plt.close(fig2)


# ==========================
# Main entry: layout and logic
# ==========================
def main() -> None:
    """
    Main entry point of the whole Streamlit app.

    Workflow:
    1. Initialize logging and page layout (wide mode).
    2. On first run:
       - Load historical configuration from config.json (if exists);
       - Map configuration into st.session_state to serve as UI defaults.
    3. Page layout:
       - Top container:
           * Left column: Environment Configuration (Game2048EnvConfig)
           * Middle column: Network Configuration (MLPConfig)
           * Right column: Agent Configuration (ReinforceAgentConfig)
       - Middle: Run mode selection (Training / Evaluation)
       - Below: run_training or run_evaluation depending on selected mode
       - Bottom: button to save current parameters as default (write to config.json)
    """
    st.set_page_config(layout="wide")

    # ========= Initialize session_state (only once, restored from config.json) =========
    if "config_inited" not in st.session_state:
        cfg = load_config()

        # Default configuration (pure data, no UI labels)
        default_cfg = {
            "env": {
                "size": 4,
                "obs_mode": "log2",
                "obs_log2_scale": 1.0,
                "reward_mode": "sum",
                "base_reward_scale": 1.0,
                "bonus_mode": "off",
                "bonus_scale": 1.0,
                "step_reward": 0.0,
                "endgame_penalty": 0.0,
                "use_action_mask": True,
                "invalid_action_penalty": -1.0,
                "max_steps": 1024,
                "empty_tile_reward": 0.0,
                "merge_reward": 0.0,
            },
            "mlp": {
                "hidden_sizes": [],
                "activation": "Sigmoid",
                "init_distribution": "XavierNormal",
                "last_init_normal": True,
            },
            "agent": {
                "gamma": 0.99,
                "learning_rate": 1e-3,
                "baseline_mode": "batch",
                "model_seed": 0,
                "reward_rank_weights": None,
                "optimizer": "sgd",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "augmentation": False,
                "use_critic": False,
                "critic_learning_rate": 1e-3,
                "critic_loss_type": "mse",
                "huber_delta": 1.0,
            },
            "train": {
                "batch_size": 32,
                "num_batches": 50,
                "env_base_seed": 12345,
                "policy_base_seed": 54321,
            },
            "eval": {
                "num_episodes": 50,
                "env_base_seed": 3,
                "policy_base_seed": 7,
                "model_path": None,
                "use_greedy": True,
            },
            "run_mode": "Training",
            "log_level": "VERBOSE",
        }

        env_cfg = cfg.get("env", {})
        mlp_cfg = cfg.get("mlp", {})
        agent_cfg = cfg.get("agent", {})
        train_cfg = cfg.get("train", {})
        eval_cfg = cfg.get("eval", {})
        run_mode = cfg.get("run_mode", default_cfg["run_mode"])
        log_level = cfg.get("log_level", default_cfg["log_level"])

        # --- Environment config -> session_state ---
        st.session_state["env_size"] = env_cfg.get("size", default_cfg["env"]["size"])
        st.session_state["obs_mode"] = env_cfg.get("obs_mode", default_cfg["env"]["obs_mode"])
        st.session_state["obs_log2_scale"] = env_cfg.get(
            "obs_log2_scale",
            default_cfg["env"]["obs_log2_scale"],
        )
        st.session_state["reward_mode"] = env_cfg.get("reward_mode", default_cfg["env"]["reward_mode"])

        st.session_state["base_reward_scale"] = env_cfg.get("base_reward_scale", default_cfg["env"]["base_reward_scale"])

        st.session_state["bonus_mode"] = env_cfg.get("bonus_mode", default_cfg["env"]["bonus_mode"])
        st.session_state["bonus_scale"] = env_cfg.get("bonus_scale", default_cfg["env"]["bonus_scale"])
        st.session_state["step_reward"] = env_cfg.get("step_reward", default_cfg["env"]["step_reward"])
        st.session_state["endgame_penalty"] = env_cfg.get(
            "endgame_penalty",
            default_cfg["env"]["endgame_penalty"],
        )

        use_action_mask_bool = env_cfg.get("use_action_mask", default_cfg["env"]["use_action_mask"])
        st.session_state["use_action_mask"] = use_action_mask_bool

        st.session_state["invalid_action_penalty"] = env_cfg.get(
            "invalid_action_penalty",
            default_cfg["env"]["invalid_action_penalty"],
        )
        st.session_state["max_steps"] = env_cfg.get("max_steps", default_cfg["env"]["max_steps"])

        st.session_state["empty_tile_reward"] = env_cfg.get(
            "empty_tile_reward", default_cfg["env"]["empty_tile_reward"]
        )
        st.session_state["merge_reward"] = env_cfg.get(
            "merge_reward", default_cfg["env"]["merge_reward"]
        )

        # --- MLP config -> session_state ---
        hidden_sizes_list = mlp_cfg.get("hidden_sizes", default_cfg["mlp"]["hidden_sizes"])
        if isinstance(hidden_sizes_list, list) and hidden_sizes_list:
            st.session_state["hidden_sizes_str"] = ",".join(str(x) for x in hidden_sizes_list)
        else:
            st.session_state["hidden_sizes_str"] = ""
        st.session_state["activation"] = mlp_cfg.get("activation", default_cfg["mlp"]["activation"])
        st.session_state["init_distribution"] = mlp_cfg.get(
            "init_distribution",
            default_cfg["mlp"]["init_distribution"],
        )
        st.session_state["last_init_normal"] = mlp_cfg.get(
            "last_init_normal",
            default_cfg["mlp"]["last_init_normal"],
        )

        # --- Agent config -> session_state ---
        st.session_state["gamma"] = agent_cfg.get("gamma", default_cfg["agent"]["gamma"])
        st.session_state["learning_rate"] = agent_cfg.get("learning_rate", default_cfg["agent"]["learning_rate"])
        st.session_state["baseline_mode"] = agent_cfg.get("baseline_mode", default_cfg["agent"]["baseline_mode"])
        st.session_state["model_seed"] = agent_cfg.get("model_seed", default_cfg["agent"]["model_seed"])

        reward_rank_weights_list = agent_cfg.get(
            "reward_rank_weights",
            default_cfg["agent"]["reward_rank_weights"],
        )
        if isinstance(reward_rank_weights_list, list) and reward_rank_weights_list:
            st.session_state["reward_rank_weights_str"] = ",".join(str(x) for x in reward_rank_weights_list)
        else:
            st.session_state["reward_rank_weights_str"] = ""

        st.session_state["optimizer"] = agent_cfg.get("optimizer", default_cfg["agent"]["optimizer"])
        st.session_state["adam_beta1"] = agent_cfg.get("adam_beta1", default_cfg["agent"]["adam_beta1"])
        st.session_state["adam_beta2"] = agent_cfg.get("adam_beta2", default_cfg["agent"]["adam_beta2"])
        st.session_state["augmentation"] = agent_cfg.get(
            "augmentation",
            default_cfg["agent"]["augmentation"],
        )
        st.session_state["use_critic"] = agent_cfg.get(
            "use_critic",
            default_cfg["agent"]["use_critic"],
        )
        st.session_state["critic_learning_rate"] = agent_cfg.get(
            "critic_learning_rate",
            default_cfg["agent"]["critic_learning_rate"],
        )
        st.session_state["critic_loss_type"] = agent_cfg.get(
            "critic_loss_type",
            default_cfg["agent"]["critic_loss_type"],
        )
        st.session_state["huber_delta"] = agent_cfg.get(
            "huber_delta",
            default_cfg["agent"]["huber_delta"],
        )

        # --- Training-related config -> session_state ---
        st.session_state["batch_size"] = train_cfg.get("batch_size", default_cfg["train"]["batch_size"])
        st.session_state["num_batches"] = train_cfg.get("num_batches", default_cfg["train"]["num_batches"])
        st.session_state["env_base_seed_train"] = train_cfg.get(
            "env_base_seed",
            default_cfg["train"]["env_base_seed"],
        )
        st.session_state["policy_base_seed_train"] = train_cfg.get(
            "policy_base_seed",
            default_cfg["train"]["policy_base_seed"],
        )

        # --- Evaluation-related config -> session_state ---
        st.session_state["num_episodes"] = eval_cfg.get("num_episodes", default_cfg["eval"]["num_episodes"])
        st.session_state["env_base_seed_eval"] = eval_cfg.get(
            "env_base_seed",
            default_cfg["eval"]["env_base_seed"],
        )
        st.session_state["policy_base_seed_eval"] = eval_cfg.get(
            "policy_base_seed",
            default_cfg["eval"]["policy_base_seed"],
        )

        raw_model_path = eval_cfg.get("model_path", default_cfg["eval"]["model_path"]),
        if raw_model_path is None or raw_model_path == "None":
            model_path_for_ui = ""
        else:
            model_path_for_ui = str(raw_model_path)
        st.session_state["eval_model_params_path"] = model_path_for_ui

        st.session_state["eval_use_greedy"] = eval_cfg.get(
            "use_greedy",
            default_cfg["eval"]["use_greedy"],
        )

        # --- Run Mode ---
        st.session_state["mode"] = run_mode

        # --- Log Level ---
        st.session_state["log_level"] = log_level

        st.session_state["config_inited"] = True

    # ========= Page title =========
    st.title("2048 REINFORCE Training & Evaluation Dashboard")

    # ========= Logging level settings =========
    st.sidebar.markdown("### Logging Settings")
    log_level_name = st.sidebar.segmented_control(
        "Log level",
        options=["DEBUG", "VERBOSE", "INFO"],
        selection_mode="single",
        width="stretch",
        key="log_level",
    )
    log_setup(log_level_name)

    # ========= Top configuration sections: Env / MLP / Agent =========
    config_container = st.container()
    with config_container:
        col1, col2, col3 = st.columns(3)

        # ====== Column 1: Environment Configuration (Game2048EnvConfig) ======
        with col1:
            st.markdown("### Environment Configuration (Game2048EnvConfig)")

            env_size = st.number_input(
                "Board size (size)",
                min_value=2,
                max_value=8,
                step=1,
                key="env_size",
            )

            obs_mode = st.segmented_control(
                "obs_mode",
                options=["raw", "log2", "onehot"],
                selection_mode="single",
                width="stretch",
                key="obs_mode",
            )

            obs_log2_scale = st.number_input(
                "obs_log2_scale (only for obs_mode='log2')",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.4f",
                key="obs_log2_scale",
            )

            reward_mode = st.segmented_control(
                "reward_mode",
                options=["sum", "log2"],
                selection_mode="single",
                width="stretch",
                key="reward_mode",
            )

            base_reward_scale = st.number_input(
                "base_reward_scale",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.4f",
                key="base_reward_scale",
            )

            bonus_mode = st.segmented_control(
                "bonus_mode",
                options=["off", "raw", "log2"],
                selection_mode="single",
                width="stretch",
                key="bonus_mode",
            )

            bonus_scale = st.number_input(
                "bonus_scale",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="bonus_scale",
            )

            empty_tile_reward = st.number_input(
                "empty_tile_reward",
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key="empty_tile_reward",
            )

            merge_reward = st.number_input(
                "merge_reward",
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key="merge_reward",
            )

            step_reward = st.number_input(
                "step_reward",
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key="step_reward",
            )

            endgame_penalty = st.number_input(
                "endgame_penalty",
                min_value=-1000.0,
                max_value=0.0,
                step=1.0,
                key="endgame_penalty",
            )

            use_action_mask = st.checkbox(
                "use_action_mask",
                key="use_action_mask",
            )

            invalid_action_penalty = st.number_input(
                "invalid_action_penalty",
                min_value=-10.0,
                max_value=0.0,
                step=0.1,
                key="invalid_action_penalty",
            )

            # max_steps: None or some fixed step value via select_slider
            options = [None] + list(range(512, 4096 + 1, 64))
            max_steps = st.select_slider(
                "max_steps",
                options=options,
                format_func=lambda x: "None" if x is None else str(x),
                key="max_steps",
            )

        # ====== Column 2: MLP Configuration (MLPConfig) ======
        with col2:
            st.markdown("### Network Configuration (MLPConfig)")

            activation = st.segmented_control(
                "activation",
                options=["Sigmoid", "ReLU"],
                selection_mode="single",
                width="stretch",
                key="activation",
            )

            init_distribution = st.segmented_control(
                "init_distribution",
                options=["XavierNormal", "HeNormal", "XavierUniform", "Normal"],
                selection_mode="single",
                width="stretch",
                key="init_distribution",
            )

            last_init_normal = st.checkbox(
                "last_init_normal (use Normal init for last layer)",
                key="last_init_normal",
            )

            hidden_sizes_str = st.text_input(
                "hidden_sizes (comma separated, e.g. 128,128)",
                help="Leave empty for no hidden layers (i.e. linear model).",
                key="hidden_sizes_str",
            )

        # ====== Column 3: Agent Configuration (ReinforceAgentConfig) ======
        with col3:
            st.markdown("### Agent Configuration (ReinforceAgentConfig)")

            gamma = st.slider(
                "gamma",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="gamma",
            )

            learning_rate = st.number_input(
                "learning_rate",
                min_value=1e-9,
                max_value=100.0,
                step=0.001,
                format="%.6f",
                key="learning_rate",
            )

            baseline_mode = st.segmented_control(
                "baseline_mode",
                options=["off", "each", "batch", "batch_norm"],
                selection_mode="single",
                width="stretch",
                key="baseline_mode",
            )

            model_seed = st.number_input(
                "model_seed",
                min_value=0,
                max_value=10_000,
                step=1,
                key="model_seed",
            )

            reward_rank_weights_str = st.text_input(
                "reward_rank_weights (comma separated floats, e.g. 3.0,2.0,1.0,1.0)",
                help="Leave empty to disable rank-based weighting (all episodes weight=1).",
                key="reward_rank_weights_str",
            )

            optimizer = st.segmented_control(
                "optimizer",
                options=["sgd", "adam"],
                selection_mode="single",
                width="stretch",
                key="optimizer",
            )

            adam_beta1 = st.number_input(
                "adam_beta1",
                min_value=0.0,
                max_value=0.9999,
                step=0.01,
                format="%.4f",
                key="adam_beta1",
                help="Only used when optimizer = 'adam'.",
            )

            adam_beta2 = st.number_input(
                "adam_beta2",
                min_value=0.0,
                max_value=0.999999,
                step=0.001,
                format="%.6f",
                key="adam_beta2",
                help="Only used when optimizer = 'adam'.",
            )

            augmentation = st.checkbox(
                "augmentation (enable data augmentation)",
                key="augmentation",
            )

            use_critic = st.checkbox(
                "use_critic (enable Critic for Actor-Critic)",
                key="use_critic",
            )

            critic_learning_rate = st.number_input(
                "critic_learning_rate",
                min_value=1e-9,
                max_value=100.0,
                step=0.001,
                format="%.6f",
                key="critic_learning_rate",
                help="Only used when use_critic = True.",
            )

            critic_loss_type = st.segmented_control(
                "critic_loss_type",
                options=["mse", "huber"],
                selection_mode="single",
                width="stretch",
                key="critic_loss_type",
                help="Loss type for critic network.",
            )

            huber_delta = st.number_input(
                "huber_delta",
                min_value=0.0,
                max_value=1000.0,
                step=0.1,
                format="%.4f",
                key="huber_delta",
                help="Delta parameter for Huber loss (used when critic_loss_type='huber').",
            )

    # ========= Parse hidden_sizes / reward_rank_weights =========
    def parse_hidden_sizes(s: str) -> List[int]:
        """
        Parse a string like "128,256,256" into [128, 256, 256].

        - If empty string, return [];
        - If parsing fails, show warning and return [].
        """
        s = s.strip()
        if not s:
            return []
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except ValueError:
            st.warning("Failed to parse hidden_sizes, using empty list [].")
            return []

    def parse_reward_rank_weights(s: str) -> list[float] | None:
        """
        Parse a string like "3.0,2.0,1.0,1.0" into [3.0, 2.0, 1.0, 1.0].

        - If empty string, return None (disable rank-based weighting);
        - If parsing fails, show warning and return None.
        """
        s = s.strip()
        if not s:
            return None
        try:
            values = [float(x.strip()) for x in s.split(",") if x.strip()]
            return values if values else None
        except ValueError:
            st.warning("Failed to parse reward_rank_weights, using None.")
            return None

    hidden_sizes = parse_hidden_sizes(hidden_sizes_str)
    reward_rank_weights = parse_reward_rank_weights(reward_rank_weights_str)

    # ========= Wrap UI values into config objects =========
    env_config = Game2048EnvConfig(
        size=env_size,
        obs_mode=obs_mode,
        obs_log2_scale=obs_log2_scale,
        reward_mode=reward_mode,
        base_reward_scale=base_reward_scale,
        bonus_mode=bonus_mode,
        bonus_scale=bonus_scale,
        step_reward=step_reward,
        endgame_penalty=endgame_penalty,
        use_action_mask=use_action_mask,
        invalid_action_penalty=invalid_action_penalty,
        max_steps=max_steps,
        empty_tile_reward=empty_tile_reward,
        merge_reward=merge_reward,
    )

    mlp_config = MLPConfig(
        hidden_sizes=hidden_sizes,
        activation=activation,
        init_distribution=init_distribution,
        last_init_normal=last_init_normal,
    )

    agent_config = ReinforceAgentConfig(
        gamma=gamma,
        learning_rate=learning_rate,
        baseline_mode=baseline_mode,
        model_seed=model_seed,
        reward_rank_weights=reward_rank_weights,
        optimizer=optimizer,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        augmentation=augmentation,
        use_critic=use_critic,
        critic_learning_rate=critic_learning_rate,
        critic_loss_type=critic_loss_type,
        huber_delta=huber_delta,
    )

    # ========= Run mode selection (Training / Evaluation) =========
    st.markdown("---")
    st.subheader("Run Mode")

    mode = st.segmented_control(
        "Select run mode",
        options=["Training", "Evaluation"],
        selection_mode="single",
        width="stretch",
        key="mode",
    )

    # ========= Create environment and agent instances =========
    env = Game2048Env(env_config)
    agent = ReinforceAgent(env, mlp_config, agent_config)

    # ========= Run according to selected mode =========
    if mode == "Training":
        run_training(env, agent, env_config, mlp_config, agent_config)
    else:
        run_evaluation(env, agent, env_config, mlp_config, agent_config)

    # ========= Save current parameters as default =========
    if st.button("Save current parameters as default"):
        cfg_to_save = {
            "env": {
                "size": env_config.size,
                "obs_mode": env_config.obs_mode,
                "obs_log2_scale": env_config.obs_log2_scale,
                "reward_mode": env_config.reward_mode,
                "base_reward_scale": env_config.base_reward_scale,
                "bonus_mode": env_config.bonus_mode,
                "bonus_scale": env_config.bonus_scale,
                "step_reward": env_config.step_reward,
                "endgame_penalty": env_config.endgame_penalty,
                "use_action_mask": env_config.use_action_mask,
                "invalid_action_penalty": env_config.invalid_action_penalty,
                "max_steps": env_config.max_steps,
                "empty_tile_reward": env_config.empty_tile_reward,
                "merge_reward": env_config.merge_reward,
            },
            "mlp": {
                "hidden_sizes": mlp_config.hidden_sizes,
                "activation": mlp_config.activation,
                "init_distribution": mlp_config.init_distribution,
                "last_init_normal": mlp_config.last_init_normal,
            },
            "agent": {
                "gamma": agent_config.gamma,
                "learning_rate": agent_config.learning_rate,
                "baseline_mode": agent_config.baseline_mode,
                "model_seed": agent_config.model_seed,
                "reward_rank_weights": agent_config.reward_rank_weights,
                "optimizer": agent_config.optimizer,
                "adam_beta1": agent_config.adam_beta1,
                "adam_beta2": agent_config.adam_beta2,
                "augmentation": agent_config.augmentation,
                "use_critic": agent_config.use_critic,
                "critic_learning_rate": agent_config.critic_learning_rate,
                "critic_loss_type": agent_config.critic_loss_type,
                "huber_delta": agent_config.huber_delta,
            },
            "train": {
                "batch_size": st.session_state.get("batch_size"),
                "num_batches": st.session_state.get("num_batches"),
                "env_base_seed": st.session_state.get("env_base_seed_train"),
                "policy_base_seed": st.session_state.get("policy_base_seed_train"),
            },
            "eval": {
                "num_episodes": st.session_state.get("num_episodes"),
                "env_base_seed": st.session_state.get("env_base_seed_eval"),
                "policy_base_seed": st.session_state.get("policy_base_seed_eval"),
                "model_path": (
                    st.session_state.get("eval_model_params_path") or None
                ),
                "use_greedy": st.session_state.get("eval_use_greedy", True),
            },
            "run_mode": mode,
            "log_level": st.session_state.get("log_level", "INFO"),
        }
        save_config(cfg_to_save)
        st.success("Saved as default configuration. It will be loaded on next startup. ✅")


# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":
    main()
