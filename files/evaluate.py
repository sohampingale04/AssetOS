"""
ASSETOS Phase 5 – Evaluation & Benchmarking
Runs DRL policy on test set, computes performance metrics,
and compares against benchmarks (equal-weight, buy-and-hold).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv

warnings.filterwarnings("ignore")


# ── Performance Metrics ───────────────────────────────────────────────────────

def compute_metrics(
    portfolio_values: np.ndarray | list,
    returns: np.ndarray | list,
    periods_per_year: int = 52,  # weekly data
    risk_free: float = 0.06,     # 6% annual (approximate India RFR)
) -> dict[str, float]:
    """
    Compute a comprehensive set of performance metrics.
    
    Parameters
    ----------
    portfolio_values : cumulative portfolio value series (starts at 1.0)
    returns          : period-by-period net returns
    """
    pv  = np.array(portfolio_values, dtype=float)
    ret = np.array(returns,          dtype=float)

    if len(ret) == 0:
        return {}

    # ── Returns ──────────────────────────────────────────────────────────────
    total_return    = (pv[-1] / pv[0]) - 1.0
    n_years         = len(ret) / periods_per_year
    ann_return      = (1.0 + total_return) ** (1.0 / max(n_years, 1e-6)) - 1.0

    # ── Risk ─────────────────────────────────────────────────────────────────
    ann_vol         = float(ret.std() * np.sqrt(periods_per_year))
    rf_per_period   = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess_ret      = ret - rf_per_period
    sharpe          = (
        float(excess_ret.mean() * periods_per_year)
        / (ann_vol + 1e-8)
    )

    # ── Sortino ──────────────────────────────────────────────────────────────
    downside_ret    = ret[ret < rf_per_period]
    downside_std    = float(downside_ret.std() * np.sqrt(periods_per_year)) if len(downside_ret) > 0 else 1e-8
    sortino         = float(excess_ret.mean() * periods_per_year) / (downside_std + 1e-8)

    # ── Max Drawdown ─────────────────────────────────────────────────────────
    peak            = np.maximum.accumulate(pv)
    drawdown        = (pv - peak) / (peak + 1e-8)
    max_dd          = float(drawdown.min())

    # ── Turnover ─────────────────────────────────────────────────────────────
    # Average absolute weight change per period (not always stored; set NaN if unavailable)
    avg_turnover    = np.nan

    return {
        "cumulative_return":  float(total_return),
        "annualised_return":  float(ann_return),
        "annualised_vol":     float(ann_vol),
        "sharpe_ratio":       float(sharpe),
        "sortino_ratio":      float(sortino),
        "max_drawdown":       float(max_dd),
        "avg_turnover":       float(avg_turnover),
    }


def compute_turnover(weight_history: list[np.ndarray]) -> float:
    if len(weight_history) < 2:
        return 0.0
    turnovers = [
        float(np.abs(weight_history[i] - weight_history[i - 1]).sum())
        for i in range(1, len(weight_history))
    ]
    return float(np.mean(turnovers))


# ── Policy Rollout ────────────────────────────────────────────────────────────

def run_policy(
    model: PPO,
    features: np.ndarray,
    asset_returns: np.ndarray,
    tickers: list[str],
    reward_weights: dict | None = None,
    max_weight: float = 0.20,
    deterministic: bool = True,
) -> dict:
    """Roll out a trained PPO model on a given data split."""
    env = PortfolioEnv(
        feature_array=features,
        asset_returns=asset_returns,
        tickers=tickers,
        reward_weights=reward_weights,
        max_weight=max_weight,
    )
    obs, _ = env.reset()

    portfolio_values  = [1.0]
    period_returns    = []
    weight_history    = [env.weights.copy()]
    all_weights       = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        period_returns.append(info["net_return"])
        weight_history.append(info["weights"])
        all_weights.append(info["weights"])

    metrics = compute_metrics(portfolio_values, period_returns)
    metrics["avg_turnover"] = compute_turnover(weight_history)

    return {
        "metrics":          metrics,
        "portfolio_values": np.array(portfolio_values),
        "period_returns":   np.array(period_returns),
        "weight_history":   np.array(all_weights),
    }


# ── Benchmarks ────────────────────────────────────────────────────────────────

def equal_weight_portfolio(
    asset_returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> dict:
    """Static equal-weight; rebalanced weekly."""
    N   = asset_returns.shape[1]
    w   = np.ones(N) / N
    pv  = [1.0]
    ret = []
    prev_w = w.copy()

    for t in range(len(asset_returns)):
        turnover   = float(np.abs(w - prev_w).sum())
        cost       = turnover * transaction_cost
        period_ret = float(np.dot(w, asset_returns[t])) - cost
        pv.append(pv[-1] * (1.0 + period_ret))
        ret.append(period_ret)
        prev_w = w.copy()

    metrics = compute_metrics(pv, ret)
    metrics["avg_turnover"] = 0.0   # static — no drift
    return {"metrics": metrics, "portfolio_values": np.array(pv), "period_returns": np.array(ret)}


def buy_and_hold_index(
    asset_returns: np.ndarray,
) -> dict:
    """Simple cap-weighted (equal here) buy-and-hold — no rebalancing."""
    N    = asset_returns.shape[1]
    w    = np.ones(N) / N
    pv   = [1.0]
    ret  = []

    for t in range(len(asset_returns)):
        period_ret = float(np.dot(w, asset_returns[t]))
        pv.append(pv[-1] * (1.0 + period_ret))
        ret.append(period_ret)
        # Drift weights — no rebalancing
        asset_vals = w * np.array([1.0 + asset_returns[t, i] for i in range(N)])
        w          = asset_vals / asset_vals.sum()

    metrics = compute_metrics(pv, ret)
    metrics["avg_turnover"] = 0.0
    return {"metrics": metrics, "portfolio_values": np.array(pv), "period_returns": np.array(ret)}


# ── Market Regime Analysis ────────────────────────────────────────────────────

def classify_regimes(portfolio_values: np.ndarray, window: int = 13) -> np.ndarray:
    """
    Label each period as bull / bear / sideways based on rolling return.
    Returns array of same length with values: 'bull', 'bear', 'sideways'.
    """
    rolling_ret = np.convolve(
        np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-8),
        np.ones(window) / window,
        mode="same",
    )
    labels = np.where(rolling_ret > 0.005, "bull",
              np.where(rolling_ret < -0.005, "bear", "sideways"))
    return labels


def regime_metrics(
    period_returns: np.ndarray,
    portfolio_values: np.ndarray,
) -> dict[str, dict]:
    labels = classify_regimes(portfolio_values)
    # Trim to returns length
    labels = labels[: len(period_returns)]

    results = {}
    for regime in ("bull", "bear", "sideways"):
        mask = labels == regime
        if mask.sum() < 5:
            continue
        sub_ret = period_returns[mask]
        sub_pv  = np.cumprod(1.0 + sub_ret)
        results[regime] = compute_metrics(sub_pv, sub_ret)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "cumulative_return": "Cumulative Return",
    "annualised_return": "Annualised Return",
    "annualised_vol":    "Annualised Volatility",
    "sharpe_ratio":      "Sharpe Ratio",
    "sortino_ratio":     "Sortino Ratio",
    "max_drawdown":      "Max Drawdown",
    "avg_turnover":      "Avg Turnover",
}


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Parameters
    ----------
    results : {strategy_name: metrics_dict}
    """
    rows = {}
    for name, m in results.items():
        row = {}
        for k, label in METRIC_LABELS.items():
            v = m.get(k, np.nan)
            if k in ("cumulative_return", "annualised_return", "annualised_vol", "max_drawdown"):
                row[label] = f"{v * 100:.2f}%" if not np.isnan(v) else "N/A"
            else:
                row[label] = f"{v:.3f}" if not np.isnan(v) else "N/A"
        rows[name] = row
    return pd.DataFrame(rows).T


# ── Plotting ──────────────────────────────────────────────────────────────────

COLOURS = {
    "DRL (PPO)":      "#1f77b4",
    "Equal Weight":   "#ff7f0e",
    "Buy & Hold":     "#2ca02c",
}

ABLATION_COLOURS = {
    "full_model":           "#1f77b4",
    "no_drawdown_penalty":  "#d62728",
    "no_transaction_cost":  "#9467bd",
    "return_only":          "#8c564b",
}


def plot_equity_curves(
    strategies: dict[str, np.ndarray],   # {name: portfolio_values array}
    title: str = "Portfolio Equity Curves",
    save_path: str = "outputs/equity_curves.png",
    date_index: pd.DatetimeIndex | None = None,
):
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    colour_map = {**COLOURS, **ABLATION_COLOURS}

    for name, pv in strategies.items():
        x   = date_index[:len(pv)] if date_index is not None else np.arange(len(pv))
        col = colour_map.get(name, None)
        ax.plot(x, pv, label=name, linewidth=1.8, color=col)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.set_xlabel("Date" if date_index is not None else "Timestep")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Equity curve saved → {save_path}")


def plot_weight_heatmap(
    weight_history: np.ndarray,   # (T, N)
    tickers: list[str],
    save_path: str = "outputs/weight_heatmap.png",
    date_index: pd.DatetimeIndex | None = None,
):
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    x = date_index[:len(weight_history)] if date_index is not None else np.arange(len(weight_history))

    bottom = np.zeros(len(weight_history))
    cmap   = plt.get_cmap("tab20")

    for i, ticker in enumerate(tickers):
        ax.fill_between(
            np.arange(len(weight_history)),
            bottom,
            bottom + weight_history[:, i],
            label=ticker.replace(".NS", ""),
            color=cmap(i / len(tickers)),
            alpha=0.85,
        )
        bottom += weight_history[:, i]

    ax.set_title("DRL Portfolio Allocation Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Timestep")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Weight heatmap saved → {save_path}")


def plot_drawdown(
    strategies: dict[str, np.ndarray],
    save_path: str = "outputs/drawdown.png",
):
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    colour_map = {**COLOURS, **ABLATION_COLOURS}

    for name, pv in strategies.items():
        peak = np.maximum.accumulate(pv)
        dd   = (pv - peak) / (peak + 1e-8)
        col  = colour_map.get(name, None)
        ax.fill_between(np.arange(len(dd)), dd, 0, alpha=0.35, color=col)
        ax.plot(dd, label=name, linewidth=1.2, color=col)

    ax.set_title("Drawdown Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Drawdown plot saved → {save_path}")


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_results(
    weight_history: np.ndarray,
    portfolio_values: np.ndarray,
    metrics: dict,
    tickers: list[str],
    date_index: pd.DatetimeIndex | None = None,
    prefix: str = "drl",
):
    os.makedirs("outputs", exist_ok=True)

    # Weights
    idx  = date_index[:len(weight_history)] if date_index is not None else np.arange(len(weight_history))
    w_df = pd.DataFrame(weight_history, index=idx, columns=[t.replace(".NS", "") for t in tickers])
    w_df.index.name = "date"
    w_df.to_csv(f"outputs/{prefix}_portfolio_allocation.csv")
    print(f"[Export] Weights → outputs/{prefix}_portfolio_allocation.csv")

    # Portfolio value
    val_idx  = date_index[:len(portfolio_values)] if date_index is not None else np.arange(len(portfolio_values))
    val_df   = pd.DataFrame({"portfolio_value": portfolio_values}, index=val_idx)
    val_df.index.name = "date"
    val_df.to_csv(f"outputs/{prefix}_portfolio_value.csv")

    # Metrics
    m_df = pd.DataFrame(metrics, index=["value"]).T.rename(columns={"value": "metric_value"})
    m_df.to_csv(f"outputs/{prefix}_performance_metrics.csv")
    print(f"[Export] Metrics → outputs/{prefix}_performance_metrics.csv")


if __name__ == "__main__":
    # Smoke test with random data
    T, N = 100, 5
    feat  = np.random.randn(T, N, 8).astype(np.float32)
    rets  = np.random.randn(T, N).astype(np.float32) * 0.01
    ticks = [f"STK{i}" for i in range(N)]

    ew  = equal_weight_portfolio(rets)
    bh  = buy_and_hold_index(rets)
    tab = metrics_table({"EW": ew["metrics"], "BH": bh["metrics"]})
    print(tab)
    print("Eval smoke test passed.")
