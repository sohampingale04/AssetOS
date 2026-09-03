"""
ASSETOS Phase 5 – Main Orchestration Script
Runs the full DRL rebalancing pipeline end-to-end.

Usage
-----
    # Full training + evaluation
    python main.py

    # Skip training, load saved model
    python main.py --skip-train --model-path models/ppo_full_final.zip

    # Run ablation study
    python main.py --ablation

    # Quick smoke-test (synthetic data, 2000 steps)
    python main.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# Local modules
from data_loader        import load_clean_prices, train_val_test_split
from feature_engineering import (
    build_feature_matrix,
    get_feature_array,
    compute_asset_returns,
    FEATURE_NAMES,
)
from train              import train, run_ablation_study, ABLATION_CONFIGS
from evaluate           import (
    run_policy,
    equal_weight_portfolio,
    buy_and_hold_index,
    metrics_table,
    plot_equity_curves,
    plot_weight_heatmap,
    plot_drawdown,
    regime_metrics,
    export_results,
)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TIMESTEPS  = 500_000
ABLATION_TIMESTEPS = 300_000
MAX_WEIGHT         = 0.20       # IPS: 20 % per asset cap


# ── Helpers ───────────────────────────────────────────────────────────────────

def slice_arrays(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    split_prices: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align feature matrix to a price split's date range,
    then return (feature_array [T,N,F], returns_array [T,N]).
    """
    idx       = features.index.intersection(split_prices.index)
    feat_sub  = features.loc[idx]
    price_sub = split_prices.loc[idx]

    feat_arr  = get_feature_array(feat_sub, tickers)
    ret_df    = compute_asset_returns(price_sub)[tickers]
    ret_arr   = ret_df.values.astype(np.float32)
    return feat_arr, ret_arr


def print_banner(text: str):
    bar = "═" * 64
    print(f"\n╔{bar}╗")
    print(f"║  {text:<62}║")
    print(f"╚{bar}╝\n")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args):

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print_banner("STEP 1 – Loading & preprocessing data")
    prices = load_clean_prices(resample_weekly=True)
    global tickers
    tickers = prices.columns.tolist()
    print(f"  Universe: {len(tickers)} stocks")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    train_prices, val_prices, test_prices = train_val_test_split(prices)

    # ── 2. Feature Engineering ─────────────────────────────────────────────
    print_banner("STEP 2 – Feature engineering")
    features, _ = build_feature_matrix(prices, drop_initial_rows=60)

    # Align each split to the feature index
    tr_feat, tr_ret   = slice_arrays(features, prices, train_prices)
    val_feat, val_ret = slice_arrays(features, prices, val_prices)
    te_feat, te_ret   = slice_arrays(features, prices, test_prices)

    print(f"  Train arrays  : {tr_feat.shape}  returns {tr_ret.shape}")
    print(f"  Val arrays    : {val_feat.shape}  returns {val_ret.shape}")
    print(f"  Test arrays   : {te_feat.shape}  returns {te_ret.shape}")

    # ── 3. Training ────────────────────────────────────────────────────────
    if args.skip_train:
        print_banner("STEP 3 – Loading pre-trained model")
        model = PPO.load(args.model_path)
        print(f"  Loaded: {args.model_path}")
    else:
        print_banner("STEP 3 – Training PPO agent")
        model = train(
            train_features=tr_feat,
            train_returns=tr_ret,
            val_features=val_feat,
            val_returns=val_ret,
            tickers=tickers,
            total_timesteps=DEFAULT_TIMESTEPS,
            run_name="ppo_full",
            max_weight=MAX_WEIGHT,
        )

    # ── 4. Test Evaluation ─────────────────────────────────────────────────
    print_banner("STEP 4 – Evaluating on test set")
    drl_result = run_policy(model, te_feat, te_ret, tickers, max_weight=MAX_WEIGHT)
    ew_result  = equal_weight_portfolio(te_ret)
    bh_result  = buy_and_hold_index(te_ret)

    results = {
        "DRL (PPO)":    drl_result["metrics"],
        "Equal Weight": ew_result["metrics"],
        "Buy & Hold":   bh_result["metrics"],
    }

    table = metrics_table(results)
    print("\n" + "─" * 75)
    print(table.to_string())
    print("─" * 75)

    # ── 5. Regime Analysis ─────────────────────────────────────────────────
    print_banner("STEP 5 – Regime analysis (bull / bear / sideways)")
    reg = regime_metrics(
        drl_result["period_returns"],
        drl_result["portfolio_values"],
    )
    for regime, m in reg.items():
        sr = m.get("sharpe_ratio", float("nan"))
        dd = m.get("max_drawdown",  float("nan"))
        print(f"  {regime:10s} → Sharpe: {sr:.3f}  MaxDD: {dd*100:.2f}%")

    # ── 6. Ablation Study ──────────────────────────────────────────────────
    if args.ablation:
        print_banner("STEP 6 – Ablation study")
        ablation_models = run_ablation_study(
            tr_feat, tr_ret, val_feat, val_ret,
            tickers=tickers,
            total_timesteps=ABLATION_TIMESTEPS,
        )

        ablation_results = {}
        ablation_pv      = {}
        for name, m in ablation_models.items():
            res = run_policy(m, te_feat, te_ret, tickers, max_weight=MAX_WEIGHT,
                             reward_weights=ABLATION_CONFIGS[name])
            ablation_results[name] = res["metrics"]
            ablation_pv[name]      = res["portfolio_values"]

        ab_table = metrics_table(ablation_results)
        print("\n── Ablation Results ──")
        print(ab_table.to_string())

        plot_equity_curves(
            ablation_pv,
            title="Ablation Study – Equity Curves (Test Set)",
            save_path="outputs/ablation_equity_curves.png",
        )

    # ── 7. Plots ───────────────────────────────────────────────────────────
    print_banner("STEP 7 – Generating plots")
    strategy_pv = {
        "DRL (PPO)":    drl_result["portfolio_values"],
        "Equal Weight": ew_result["portfolio_values"],
        "Buy & Hold":   bh_result["portfolio_values"],
    }

    test_dates = features.index.intersection(test_prices.index)

    plot_equity_curves(
        strategy_pv,
        title="ASSETOS Phase 5 – Test Set Equity Curves",
        save_path="outputs/equity_curves.png",
        date_index=test_dates if len(test_dates) >= len(drl_result["portfolio_values"]) else None,
    )
    plot_drawdown(strategy_pv, save_path="outputs/drawdown.png")
    plot_weight_heatmap(
        drl_result["weight_history"],
        tickers=tickers,
        save_path="outputs/weight_heatmap.png",
    )

    # ── 8. Export CSV ──────────────────────────────────────────────────────
    print_banner("STEP 8 – Exporting CSV outputs")
    export_results(
        weight_history=drl_result["weight_history"],
        portfolio_values=drl_result["portfolio_values"],
        metrics=drl_result["metrics"],
        tickers=tickers,
        prefix="drl",
    )

    # Save benchmark metrics too
    bm_metrics = pd.DataFrame({
        "DRL (PPO)":    drl_result["metrics"],
        "Equal Weight": ew_result["metrics"],
        "Buy & Hold":   bh_result["metrics"],
    }).T
    bm_metrics.index.name = "strategy"
    bm_metrics.to_csv("outputs/performance_metrics.csv")
    print("[Export] outputs/performance_metrics.csv")

    print_banner("✅  Phase 5 complete")
    print("  Outputs:")
    for f in sorted(Path("outputs").glob("*")):
        print(f"    {f}")


# ── Smoke Test ────────────────────────────────────────────────────────────────

def run_smoke_test():
    """Fully synthetic end-to-end test — no network needed."""
    print_banner("SMOKE TEST – Synthetic data")
    global tickers
    T_tr, T_val, T_te, N, F = 300, 60, 80, 6, 8
    tickers = [f"STK{i}" for i in range(N)]

    tr_feat  = np.random.randn(T_tr,  N, F).astype(np.float32)
    tr_ret   = np.random.randn(T_tr,  N).astype(np.float32) * 0.01
    val_feat = np.random.randn(T_val, N, F).astype(np.float32)
    val_ret  = np.random.randn(T_val, N).astype(np.float32) * 0.01
    te_feat  = np.random.randn(T_te,  N, F).astype(np.float32)
    te_ret   = np.random.randn(T_te,  N).astype(np.float32) * 0.01

    model = train(
        tr_feat, tr_ret, val_feat, val_ret,
        tickers=tickers,
        total_timesteps=2_000,
        run_name="smoke_test",
        max_weight=0.25,
    )

    drl = run_policy(model, te_feat, te_ret, tickers, max_weight=0.25)
    ew  = equal_weight_portfolio(te_ret)
    bh  = buy_and_hold_index(te_ret)

    table = metrics_table({
        "DRL (PPO)":    drl["metrics"],
        "Equal Weight": ew["metrics"],
        "Buy & Hold":   bh["metrics"],
    })
    print(table.to_string())

    plot_equity_curves(
        {"DRL (PPO)": drl["portfolio_values"], "EW": ew["portfolio_values"]},
        save_path="outputs/smoke_equity.png",
    )
    plot_drawdown(
        {"DRL (PPO)": drl["portfolio_values"], "EW": ew["portfolio_values"]},
        save_path="outputs/smoke_drawdown.png",
    )
    plot_weight_heatmap(
        drl["weight_history"], tickers=tickers,
        save_path="outputs/smoke_weights.png",
    )
    export_results(
        drl["weight_history"], drl["portfolio_values"],
        drl["metrics"], tickers, prefix="smoke"
    )
    print_banner("✅  Smoke test passed")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ASSETOS Phase 5 – DRL Rebalancing")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training; load model from --model-path")
    p.add_argument("--model-path",  type=str,
                   default="models/ppo_full_final.zip",
                   help="Path to saved .zip model (used with --skip-train)")
    p.add_argument("--ablation",    action="store_true",
                   help="Run the full ablation study after main training")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick end-to-end test with synthetic data")
    p.add_argument("--timesteps",   type=int, default=DEFAULT_TIMESTEPS,
                   help=f"PPO training timesteps (default: {DEFAULT_TIMESTEPS:,})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("data",    exist_ok=True)
    os.makedirs("logs",    exist_ok=True)

    if args.timesteps != DEFAULT_TIMESTEPS:
        DEFAULT_TIMESTEPS = args.timesteps

    if args.smoke_test:
        run_smoke_test()
    else:
        run_pipeline(args)
