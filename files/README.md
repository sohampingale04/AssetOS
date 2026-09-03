# ASSETOS – Phase 5: DRL Portfolio Rebalancing

Deep Reinforcement Learning system using **PPO** (Proximal Policy Optimisation)
that learns optimal portfolio rebalancing strategies for a NIFTY stock universe.

---

## Project Structure

```
assetos_phase5/
├── data_loader.py          # Download NIFTY data via yfinance, clean & cache
├── feature_engineering.py  # Returns, volatility, momentum + rolling z-score
├── portfolio_env.py        # Custom Gymnasium environment (PortfolioEnv)
├── train.py                # PPO model builder, training loop, ablation configs
├── evaluate.py             # Metrics, benchmarks, plots, CSV export
├── main.py                 # End-to-end orchestration + CLI
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1 – Smoke test (synthetic data, no internet needed)
```bash
python main.py --smoke-test
```

### 2 – Full pipeline (downloads real NIFTY data)
```bash
python main.py
```

### 3 – Full pipeline + ablation study
```bash
python main.py --ablation
```

### 4 – Skip training, load saved model
```bash
python main.py --skip-train --model-path models/ppo_full_final.zip
```

### 5 – Custom training timesteps
```bash
python main.py --timesteps 1000000
```

---

## Pipeline Steps

| # | Step | Module |
|---|------|--------|
| 1 | Download & clean NIFTY weekly prices | `data_loader.py` |
| 2 | Feature engineering + rolling z-score | `feature_engineering.py` |
| 3 | Train PPO on 2010–2021 | `train.py` |
| 4 | Evaluate on 2023–present | `evaluate.py` |
| 5 | Regime analysis (bull/bear/sideways) | `evaluate.py` |
| 6 | Ablation study (optional) | `train.py` |
| 7 | Plots (equity curves, drawdown, weights) | `evaluate.py` |
| 8 | Export CSVs | `evaluate.py` |

---

## Environment Details (`PortfolioEnv`)

| Component | Detail |
|-----------|--------|
| **State** | Return history (20 × N) + features (N × F) + current weights (N) + 4 portfolio scalars |
| **Action** | Raw logits → projected onto constrained simplex (softmax + per-asset cap) |
| **Reward** | `1.0 × ret − 0.5 × vol − 0.1 × turnover − 0.3 × drawdown` |
| **Transaction cost** | 0.1% per unit of turnover |
| **Max per-asset weight** | 20% (IPS constraint) |

---

## Benchmarks

The DRL strategy is compared against:
- **Equal-weight portfolio** (rebalanced weekly)
- **Buy-and-hold index** (no rebalancing)

---

## Ablation Variants

| Variant | Change |
|---------|--------|
| `full_model` | All reward components active |
| `no_drawdown_penalty` | Drawdown coefficient = 0 |
| `no_transaction_cost` | Turnover coefficient = 0 |
| `return_only` | Only return term active |

---

## Outputs

All outputs are written to `outputs/`:

| File | Description |
|------|-------------|
| `drl_portfolio_allocation.csv` | Time-series of portfolio weights |
| `drl_portfolio_value.csv` | Cumulative portfolio value |
| `performance_metrics.csv` | All strategy metrics (Sharpe, Sortino, MDD …) |
| `equity_curves.png` | DRL vs benchmarks equity curves |
| `drawdown.png` | Drawdown comparison |
| `weight_heatmap.png` | Stacked-area allocation chart |
| `ablation_equity_curves.png` | Ablation comparison (if `--ablation`) |

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Tracks: episode reward, portfolio value, turnover, drawdown per step.

---

## Data Split

| Split | Period |
|-------|--------|
| Train | 2010 – end 2021 |
| Validation | 2022 |
| Test | 2023 – latest |

> **Important:** No random shuffling — strict temporal ordering maintained.

---

## Success Criteria

- ✅ Sharpe ratio > equal-weight baseline on test set
- ✅ Lower max drawdown than buy-and-hold
- ✅ Controlled turnover (< 0.5 per period average)
