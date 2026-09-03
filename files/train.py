"""
ASSETOS Phase 5 – PPO Training
Trains a Proximal Policy Optimisation agent on the PortfolioEnv.
Supports ablation study by passing custom reward_weights.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from portfolio_env import PortfolioEnv


# ── Logging Callback ──────────────────────────────────────────────────────────

class PortfolioMetricsCallback(BaseCallback):
    """
    Logs custom portfolio metrics (portfolio value, turnover) to TensorBoard
    at the end of every episode.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_turnover: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "portfolio_value" in info:
                self.logger.record(
                    "portfolio/value", info["portfolio_value"]
                )
                self.logger.record(
                    "portfolio/turnover", info["turnover"]
                )
                self.logger.record(
                    "portfolio/net_return", info["net_return"]
                )
                self.logger.record(
                    "portfolio/drawdown", info["drawdown"]
                )
        return True


# ── Environment Factory ───────────────────────────────────────────────────────

def make_env(
    feature_array: np.ndarray,
    asset_returns: np.ndarray,
    tickers: list[str],
    reward_weights: dict | None = None,
    max_weight: float = 0.20,
    seed: int = 42,
):
    """Returns a callable that creates a monitored PortfolioEnv."""
    def _init():
        env = PortfolioEnv(
            feature_array=feature_array,
            asset_returns=asset_returns,
            tickers=tickers,
            reward_weights=reward_weights,
            max_weight=max_weight,
        )
        env = Monitor(env)
        return env
    return _init


# ── PPO Builder ───────────────────────────────────────────────────────────────

def build_ppo(
    env: DummyVecEnv,
    tensorboard_log: str = "logs/tensorboard",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    policy_kwargs: dict | None = None,
    seed: int = 42,
) -> PPO:
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
    )
    return model


# ── Main Training Function ────────────────────────────────────────────────────

def train(
    train_features: np.ndarray,
    train_returns: np.ndarray,
    val_features: np.ndarray,
    val_returns: np.ndarray,
    tickers: list[str],
    total_timesteps: int = 500_000,
    reward_weights: dict | None = None,
    run_name: str = "ppo_full",
    save_dir: str = "models",
    max_weight: float = 0.20,
    seed: int = 42,
) -> PPO:
    """
    Train a PPO agent.

    Parameters
    ----------
    train_features / train_returns : train split arrays
    val_features / val_returns     : validation split arrays
    tickers                        : list of asset names
    total_timesteps                : number of env steps to train for
    reward_weights                 : dict override for ablation study
    run_name                       : used for model save file and TB log
    save_dir                       : directory for saved models
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    # ── Training environment ─────────────────────────────────────────────────
    train_env = DummyVecEnv([
        make_env(train_features, train_returns, tickers,
                 reward_weights=reward_weights, max_weight=max_weight, seed=seed)
    ])

    # ── Validation environment ───────────────────────────────────────────────
    val_env = DummyVecEnv([
        make_env(val_features, val_returns, tickers,
                 reward_weights=reward_weights, max_weight=max_weight, seed=seed + 1)
    ])

    # ── Callbacks ────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env=val_env,
        best_model_save_path=f"{save_dir}/{run_name}_best",
        log_path=f"logs/{run_name}_eval",
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"{save_dir}/checkpoints/{run_name}",
        name_prefix="ppo",
    )
    metrics_callback = PortfolioMetricsCallback()

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_ppo(
        env=train_env,
        tensorboard_log=f"logs/tensorboard/{run_name}",
        seed=seed,
    )

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Assets          : {len(tickers)}")
    print(f"  Reward weights  : {reward_weights or 'default'}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, metrics_callback],
        tb_log_name=run_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # Save final model
    save_path = f"{save_dir}/{run_name}_final"
    model.save(save_path)
    print(f"[Train] Model saved → {save_path}.zip")

    # Save run config
    config = {
        "run_name":         run_name,
        "total_timesteps":  total_timesteps,
        "tickers":          tickers,
        "reward_weights":   reward_weights,
        "max_weight":       max_weight,
        "seed":             seed,
    }
    with open(f"{save_dir}/{run_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model


# ── Ablation Configs ──────────────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict | None] = {
    "full_model": None,  # default weights

    "no_drawdown_penalty": {
        "return":    1.0,
        "vol":      -0.5,
        "turnover": -0.1,
        "drawdown":  0.0,   # disabled
    },

    "no_transaction_cost": {
        "return":    1.0,
        "vol":      -0.5,
        "turnover":  0.0,   # disabled
        "drawdown": -0.3,
    },

    "return_only": {
        "return":    1.0,
        "vol":       0.0,   # disabled
        "turnover":  0.0,   # disabled
        "drawdown":  0.0,   # disabled
    },
}


def run_ablation_study(
    train_features: np.ndarray,
    train_returns: np.ndarray,
    val_features: np.ndarray,
    val_returns: np.ndarray,
    tickers: list[str],
    total_timesteps: int = 300_000,
    save_dir: str = "models/ablation",
) -> dict[str, PPO]:
    """Train all ablation variants and return trained models."""
    trained = {}
    for name, rw in ABLATION_CONFIGS.items():
        print(f"\n>>> Ablation: {name}")
        model = train(
            train_features=train_features,
            train_returns=train_returns,
            val_features=val_features,
            val_returns=val_returns,
            tickers=tickers,
            total_timesteps=total_timesteps,
            reward_weights=rw,
            run_name=name,
            save_dir=save_dir,
        )
        trained[name] = model
    return trained


if __name__ == "__main__":
    # Quick smoke test
    T_tr, T_val, N, F = 300, 60, 5, 8
    tr_feat  = np.random.randn(T_tr,  N, F).astype(np.float32)
    tr_ret   = np.random.randn(T_tr,  N).astype(np.float32) * 0.01
    val_feat = np.random.randn(T_val, N, F).astype(np.float32)
    val_ret  = np.random.randn(T_val, N).astype(np.float32) * 0.01
    tickers  = [f"STK{i}" for i in range(N)]

    model = train(
        tr_feat, tr_ret, val_feat, val_ret, tickers,
        total_timesteps=2000,
        run_name="smoke_test",
    )
    print("Training smoke test passed.")
