# 🏦 AssetOS — Intelligent Portfolio System (IPS)

> **An institutional-grade, full-stack quantitative portfolio engine for the Indian equity market (Nifty 100 universe), combining classical financial theory (Markowitz MPT, Black-Litterman), Reinforcement Learning (Recurrent PPO with LSTM), advanced stress testing, and a premium React web interface.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Tech Stack](#4-tech-stack)
5. [Core Engine — IPS.py](#5-core-engine--ipspy)
6. [Financial Models Deep Dive](#6-financial-models-deep-dive)
7. [Reinforcement Learning — PPO with LSTM](#7-reinforcement-learning--ppo-with-lstm)
8. [Institutional Stress Testing Engine](#8-institutional-stress-testing-engine)
9. [Backend API — Flask REST Server](#9-backend-api--flask-rest-server)
10. [Frontend — React Web Interface](#10-frontend--react-web-interface)
11. [Data Layer](#11-data-layer)
12. [Data Diagnostic Tool](#12-data-diagnostic-tool)
13. [Installation & Setup](#13-installation--setup)
14. [Run Commands](#14-run-commands)
15. [API Reference](#15-api-reference)
16. [Key Configuration](#16-key-configuration)
17. [Logging](#17-logging)
18. [Saved Model Artifacts](#18-saved-model-artifacts)

---

## 1. Project Overview

**AssetOS** is a full-stack quantitative fintech platform built around the **Nifty 100** Indian equity universe. The project demonstrates how modern AI and classical finance can coexist:

- **Classical Finance Layer**: Markowitz Mean-Variance Optimization and the Black-Litterman model provide the theoretical backbone.
- **AI / RL Layer**: A Recurrent PPO (Proximal Policy Optimization) agent with an LSTM policy, trained using the `stable-baselines3` / `sb3-contrib` libraries, learns to dynamically rebalance the portfolio in response to evolving market conditions.
- **Stress Testing Layer**: An institutional-grade engine evaluates portfolio robustness against 8 macro-economic scenarios (GFC 2008, COVID-2020, Stagflation, etc.) using Extreme Value Theory (EVT) and tail-risk analytics.
- **Frontend Layer**: A premium React (Vite + TailwindCSS + Recharts + Framer Motion) dashboard visualizes every output — allocation pie charts, Monte Carlo cones, RL training curves, and stress test reports.

The entire workflow starts from an investor onboarding flow and culminates in a personalized, risk-adjusted portfolio allocation backed by both math and machine learning.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│            React Frontend (Vite, Port: 5173)                 │
│                                                              │
│  Onboarding → Dashboard → MarketViews → RLRebalancingEngine  │
│  (TailwindCSS v4, Recharts, Framer Motion, Lucide Icons)     │
└─────────────────────────┬────────────────────────────────────┘
                          │  HTTP REST / JSON
┌─────────────────────────▼────────────────────────────────────┐
│            Flask Backend API (Port: 5000)                    │
│  backend/app.py  ← CORS-enabled Flask REST server            │
│                                                              │
│  /api/onboard            → Investor profiling                │
│  /api/generate_portfolio → MPT optimization + Monte Carlo    │
│  /api/optimize_bl        → Black-Litterman + Monte Carlo     │
│  /api/rl_rebalance       → RL-driven regime-aware rebalance  │
│  /api/resimulate         → Re-run Monte Carlo (same weights) │
│  /api/assets             → List available tickers            │
└─────────────────────────┬────────────────────────────────────┘
                          │  Python Imports
┌─────────────────────────▼────────────────────────────────────┐
│         Core Quantitative Engine (IPS.py)                    │
│                                                              │
│  PortfolioBuilder class                                      │
│  ├── run_onboarding()       Investor interview (CLI)         │
│  ├── run_api_onboarding()   API-compatible intake            │
│  ├── generate_ips()         Risk scoring + IPS constraints   │
│  ├── load_market_data()     CSV → log returns → mu, Sigma    │
│  ├── _optimize()            SLSQP Max-Sharpe optimizer       │
│  ├── run_black_litterman()  BL master formula                │
│  ├── run_mpt_only()         Wrapper: pure MPT                │
│  ├── run_full_simulation()  1000-path regime-switching MC    │
│  └── run_monte_carlo()      Legacy: simpler 400-path MC      │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌────────────────┬─────────▼────────────────┬──────────────────┐
│  stress_       │   PPO_training.ipynb     │  Data Layer      │
│  testing.py    │   (Offline RL training)  │                  │
│                │                          │  NIFTY100_       │
│  8 Macro       │   RecurrentPPO + LSTM    │  Sectors_Initial │
│  Scenarios     │   stable-baselines3      │  *.csv           │
│  EVT / CVaR    │   sb3-contrib            │                  │
│  Sensitivity   │   gymnasium              │  AssetOS_Data/   │
│  IPS Violation │   yfinance               │  *.csv           │
└────────────────┴──────────────────────────┴──────────────────┘
```

---

## 3. Project Structure

```
Major_Project/
│
├── IPS.py                          # Core quantitative engine (PortfolioBuilder, 994 lines)
├── investor_profiler.py            # CSV-based investor profile reader & constraint generator
├── Diagnostic.py                   # Data health checker — risk premium + correlation audit
├── stress_testing.py               # Institutional stress testing engine (1513 lines)
├── system_validator.py             # System health validator
├── run_gui.py                      # One-click launcher: starts backend + frontend
├── app.py                          # Legacy simple Flask wrapper (demo only)
├── test_bl.py                      # Isolated Black-Litterman unit test
├── test.py                         # General test script
│
├── PPO_training.ipynb              # Full RL training notebook (RecurrentPPO + LSTM)
├── Major_Code.ipynb                # Primary research & exploration notebook
├── Rebalancing.ipynb               # Portfolio rebalancing experiments
├── Rebalancing_Fixed.ipynb         # Fixed/improved rebalancing experiments
│
├── ppo_antioverfitting.zip         # Trained PPO model (anti-overfit variant, ~7.4MB)
├── ppo_fixed.zip                   # Trained PPO model (fixed variant, ~1.5MB)
├── ppo_lstm_portfolio.zip          # Trained PPO model (LSTM variant, ~3.0MB)
├── ppo_portfolio_nse.zip           # Trained PPO model (NSE full universe, ~4.1MB)
├── ppo_portfolio_rebalancer.zip    # Trained PPO model (rebalancer, ~747KB)
│
├── vecnorm_antioverfitting.pkl     # VecNormalize stats for anti-overfit model (~47KB)
├── vecnorm_fixed.pkl               # VecNormalize stats for fixed model (~7KB)
├── vecnorm_nse.pkl                 # VecNormalize stats for NSE model (~8KB)
│
├── combined_nifty100_prices.csv    # Merged Nifty100 price data (~290KB)
├── portfolio_allocation.csv        # Latest generated portfolio allocation (~39KB)
├── reliance.csv                    # Reliance Industries price history
├── portfolio_performance.png       # Portfolio performance chart
├── test_set.png                    # RL agent test-set performance chart
├── validation_set.png              # RL agent validation-set performance chart
│
├── asset_os.log                    # Runtime log file (all API + optimization events)
│
├── backend/
│   └── app.py                      # Flask REST API (368 lines, 7 endpoints)
│
├── frontend/
│   ├── index.html                  # Root HTML entry point
│   ├── package.json                # Node dependencies
│   ├── vite.config.js              # Vite build config with proxy to Flask
│   ├── tailwind.config.js          # TailwindCSS v4 config (custom design tokens)
│   ├── postcss.config.js           # PostCSS config
│   └── src/
│       ├── main.jsx                # React entry point
│       ├── App.jsx                 # Root component + page routing
│       ├── index.css               # Global styles, CSS variables, design system
│       └── components/
│           ├── Onboarding.jsx      # Investor intake flow (14KB)
│           ├── Dashboard.jsx       # Portfolio charts + risk analysis (27KB)
│           ├── MarketViews.jsx     # Black-Litterman views input (8KB)
│           └── RLRebalancingEngine.jsx  # RL rebalancer UI (22KB)
│
├── NIFTY100_Sectors_Initial/       # Primary stock price CSVs (sector-organized)
│   └── *.csv                       # Individual stock price histories (Adj Close)
│
├── AssetOS_Data/                   # Additional data and profiles
│   ├── *.csv                       # Individual stock price histories
│   └── investor_profile.csv        # Investor risk profile dataset
│
├── drl_tensorboard/                # TensorBoard logs for base DRL training
├── drl_tensorboard_lstm/           # TensorBoard logs for LSTM DRL training
├── stress_reports/                 # Generated stress testing report files
└── validation_results/             # Validation outputs and charts
```

---

## 4. Tech Stack

### Backend (Python)

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.12.x | Core language |
| `Flask` | Latest | REST API server |
| `flask-cors` | Latest | Cross-Origin Resource Sharing for React |
| `numpy` | Latest | Matrix algebra, vectorized simulation |
| `pandas` | Latest | Data wrangling, time-series alignment |
| `scipy` | Latest | SLSQP optimizer, EVT (genpareto), Student-t |
| `matplotlib` | Latest | Monte Carlo cone plots, RL performance charts |
| `seaborn` | Latest | Heatmaps and financial visualization |
| `yfinance` | Latest | Historical price data download (RL training) |
| `gymnasium` | Latest | RL environment standard (OpenAI Gym successor) |
| `stable-baselines3` | Latest | PPO algorithm implementation |
| `sb3-contrib` | Latest | RecurrentPPO (PPO + LSTM) implementation |
| `pytz` | Latest | Timezone-aware datetime handling |

### Frontend (JavaScript/React)

| Library | Version | Purpose |
|---|---|---|
| `React` | v19.2.0 | UI framework |
| `Vite` | v7.3.1 | Dev server & production bundler |
| `TailwindCSS` | v4.1.18 | Utility-first CSS with custom design tokens |
| `Recharts` | v3.7.0 | Portfolio pie charts, Monte Carlo area charts, RL training charts |
| `Framer Motion` | v12.34.0 | Page transitions, card animations, micro-interactions |
| `Lucide React` | v0.564.0 | Icon library |
| `PostCSS` | v8.5.6 | CSS transformation pipeline |
| `ESLint` | v9.39.1 | Code linting |

### Machine Learning / RL

| Tool | Purpose |
|---|---|
| `stable-baselines3` | PPO base implementation |
| `sb3-contrib` | RecurrentPPO with LSTM (stateful policy) |
| `gymnasium` | RL environment (state space, action space, reward) |
| `VecNormalize` | Running normalization of observations and rewards |
| `DummyVecEnv` | Parallel environment wrapper for SB3 |
| `TensorBoard` | Training visualization (reward, KL divergence, policy loss) |

---

## 5. Core Engine — IPS.py

`IPS.py` is the **brain** of the entire project. It contains the `PortfolioBuilder` class — 994 lines of financial math and simulation logic.

### Global Configuration

```python
CONFIG = {
    'data_dir': "NIFTY100_Sectors_Initial",
    'risk_free_rate': 0.065,   # India 10Y government gilt proxy
    'log_file': "asset_os.log",
    'trading_days': 252        # NSE trading calendar
}
```

### `PortfolioBuilder` Class — Method Summary

| Method | Description |
|---|---|
| `run_onboarding()` | Interactive CLI client interview (10 parameters) |
| `run_api_onboarding(data)` | API-compatible version — accepts dict directly |
| `generate_ips()` | Computes composite risk score → generates IPS constraints |
| `load_market_data()` | Reads CSVs, aligns dates, computes log returns, annualizes mu and Sigma |
| `_optimize(mu, cov, constraints)` | SLSQP optimizer — maximizes Sharpe Ratio |
| `run_mpt_only()` | Wrapper: pure MPT optimization without views |
| `run_black_litterman(views_map)` | Full BL formula with user-defined views |
| `run_monte_carlo(allocation)` | Legacy 400-path GBM simulation with plot |
| `run_full_simulation(allocation)` | Full 1000-path regime-switching MC with 5 percentile bands, scorecard, benchmarks |
| `get_monte_carlo_data(allocation)` | Legacy alias for backward compatibility |

---

## 6. Financial Models Deep Dive

### 6.1 Investor Profiling & IPS Generation

The `generate_ips()` method implements a **composite risk scoring model**. Every onboarding parameter meaningfully contributes to the final Investment Policy Statement.

#### Composite Risk Score Formula

```
Risk Score = (attitude/5 × 0.35)           [Risk Attitude: 35%]
           + (horizon_factor × 0.35)        [Time Horizon: 35%]
           + ((1 - liquidity) × 0.15)       [Liquidity Need: 15%]
           + (goal_modifier × 0.15)         [Goal Type: 15%]

Risk Score is clamped to [0.05, 0.95]
```

**Horizon Factor** is a non-linear piecewise function:

| Horizon | Horizon Factor |
|---|---|
| ≤ 2 years | 0.10 (very conservative) |
| 2–5 years | 0.30 to 0.60 (linear) |
| 5–10 years | 0.60 to 0.85 (linear) |
| > 10 years | 0.85+ (capped at 1.0) |

**Goal Modifier:**
- Aggressive/Growth → +0.15
- Preservation/Conservative → -0.15
- Income/Balanced → 0.0

#### IPS Output Parameters

| Parameter | Derivation |
|---|---|
| `ReturnObjective` | `6.5% + (risk_score × 12%)`, ±3% goal adjustment |
| `EquityMin / EquityMax` | 6-tier bracket from risk_score (10%–100%) |
| `CashMin / CashMax` | Directly from liquidity input; raised floor for short horizons |
| `MaxDrawdown` | Directly from loss_tolerance input |

**Age-based hard overrides:**
- Age < 30 → `EquityMax + 10%`
- Age > 60 → `EquityMax - 15%`
- Horizon ≤ 2 years → hard cap `EquityMax = 40%`

---

### 6.2 Mean-Variance Optimization (MPT)

AssetOS implements the classical **Markowitz Efficient Frontier** approach, maximizing the Sharpe Ratio subject to IPS constraints.

#### Returns & Covariance Calculation

Data is processed as **log returns** (preferred for optimization math):

```
r_t = ln(P_t / P_{t-1})

Annualized:
  mu  = mean(r) × 252
  Cov = cov(r)  × 252
```

#### Optimization Problem

```
Maximize:  S(w) = (w^T * mu - r_f) / sqrt(w^T * Cov * w)

Subject to:
  EquityMin <= sum(w_i)       <= EquityMax
  CashMin   <= 1 - sum(w_i)  <= CashMax
  w_i >= 0 for all i
```

The objective function also includes **penalty terms** to push the portfolio toward IPS targets:

```
Objective = -S(w)
          + 10 * max(0, target_return - portfolio_return)^2
          +  5 * max(0, portfolio_vol - max_drawdown)^2
```

**Solver**: `scipy.optimize.minimize` with method `SLSQP` (Sequential Least Squares Programming).

- Initial guess: uniform equity distribution at `EquityMin`
- Cash is set dynamically as the residual: `CASH_RESERVE = 1 - sum(w_i)`

**Synthetic assets** added to the universe as diversifiers:

| Asset | Return | Volatility |
|---|---|---|
| `BOND_INDEX` | 7.5% | 5% |
| `ALT_INDEX` | 9.0% | 10% |

---

### 6.3 Black-Litterman Model

The Black-Litterman model allows users to **inject subjective views** on specific stocks, blending them mathematically with the market equilibrium.

#### Parameters

| Symbol | Value | Meaning |
|---|---|---|
| delta (δ) | 2.5 | Risk aversion coefficient |
| tau (τ) | 0.05 | Scaling factor (uncertainty in equilibrium) |

#### The BL Master Equation

```
mu_BL = [ (tau*Sigma)^-1  +  P^T * Omega^-1 * P ]^-1
        [ (tau*Sigma)^-1 * Pi  +  P^T * Omega^-1 * Q ]
```

**Step-by-step implementation:**

1. **Market Weights `w`**: Equal-weight proxy for the Nifty 100 equilibrium.

2. **Implied Equilibrium Returns (reverse-optimized from market portfolio):**
   ```
   Pi = delta × Sigma × w
   ```

3. **Pick Matrix `P`** *(n_views × n_assets)*: Absolute views — one row per stock, value 1 in the relevant column.

4. **View Vector `Q`** *(n_views × 1)*: User's expected returns (e.g., 0.25 = +25% for Bullish).

5. **View Uncertainty Matrix `Omega`** *(diagonal)*:
   ```
   Omega = diag( P × (tau × Sigma) × P^T )
   ```
   Ties uncertainty to the statistical confidence of the historical covariance.

6. **BL Expected Returns**: Computed via `numpy.linalg.pinv` (pseudo-inverse) for numerical stability against singular matrices.

7. **Re-optimization**: SLSQP optimizer re-run with `mu_BL` replacing historical `mu`, with the original `Sigma` unchanged (views change return expectations, not risk).

**User View Sentiments:**

| Sentiment | Return Injected |
|---|---|
| Bullish | +25% expected return |
| Neutral | +12% expected return |
| Bearish | −5% expected return |

---

### 6.4 Monte Carlo Simulation (Cone of Uncertainty)

AssetOS runs **1,000 Geometric Brownian Motion paths** with a **regime-switching volatility** layer over the user's entire investment horizon (1–30 years).

#### GBM Formula

```
S(t) = S(t-1) × exp( (mu_d - 0.5×sigma_d^2) + sigma_adj × Z_t )

Where:
  mu_d      = mu_port / 252           (daily drift)
  sigma_d   = sigma_port / sqrt(252)  (daily volatility)
  Z_t ~ N(0,1)                        (standard normal shock)
  sigma_adj = regime-adjusted volatility
```

#### Regime-Switching Volatility (3-State Markov Chain)

The simulation uses a **Hidden Markov Model** with 3 volatility regimes that evolve day-by-day:

| Regime | Vol Multiplier | Stay Prob | → Calm | → Normal | → Stressed |
|---|---|---|---|---|---|
| Calm (0) | 0.7× base vol | 97.0% | — | 2.5% | 0.5% |
| Normal (1) | 1.0× base vol | 97.0% | 1.5% | — | 1.5% |
| Stressed (2) | 1.8× base vol | 95.0% | 1.0% | 4.0% | — |

This captures **volatility clustering** (stressed periods tend to persist), making the simulation more realistic than constant-volatility GBM.

#### Cash Portion (Grows at Risk-Free Rate Separately)

```
Cash(t) = W_0 × w_cash × (1 + r_f)^(t/252)
```

#### Output Percentile Bands

| Band | Percentile | Meaning |
|---|---|---|
| P5 | 5th | Extreme bear case |
| P25 | 25th | Conservative case |
| P50 | 50th | Median / most likely |
| P75 | 75th | Optimistic case |
| P95 | 95th | Extreme bull case |

Monthly snapshots (every 21 trading days) are returned for the frontend chart.

---

### 6.5 Portfolio Scorecard & Risk Metrics

The `run_full_simulation()` method computes a comprehensive set of risk metrics and a graded scorecard.

#### Computed Metrics

| Metric | Formula / Method |
|---|---|
| **Sharpe Ratio** | `(r_p - r_f) / sigma_p` |
| **Sortino Ratio** | `r_p / sigma_downside` (downside deviation per path, mean across paths) |
| **Max Drawdown** | Median of per-path: `max((peak - value) / peak)` |
| **Diversification Ratio** | `sum(w_i × sigma_i) / sigma_p` (weighted individual vols / portfolio vol) |
| **VaR 95%** | Annualized 5th percentile of daily returns × sqrt(252) |
| **CVaR 95%** | Mean of daily returns below VaR threshold, annualized |
| **Liquidity Score** | `min(1.0, w_cash × 5)` — scaled 0–100 |
| **Goal Probability** | % of simulation paths exceeding `capital × (1 + r_target)^horizon` |

#### Portfolio Scorecard (A+ to D grades)

| Dimension | Weight | Source |
|---|---|---|
| Risk-Adjusted Return | 30% | Sharpe Ratio → scaled 0–100 |
| Downside Protection | 25% | Max Drawdown → penalized score |
| Diversification | 15% | Diversification Ratio |
| Liquidity | 10% | Cash allocation score |
| Goal Alignment | 20% | Portfolio return vs. IPS target return |

Overall score = weighted sum → letter grade (A+ ≥ 90, A ≥ 80, A- ≥ 75, B+ ≥ 70, B ≥ 60, C ≥ 50, D < 50).

#### Benchmarks Computed

| Benchmark | Parameters |
|---|---|
| Nifty 50 proxy | 12% return, 18% vol |
| Fixed Deposit | 6.5% annual (risk-free rate) |
| Savings Account | 3.5% annual |

---

## 7. Reinforcement Learning — PPO with LSTM

The RL component is implemented in `PPO_training.ipynb`. It trains an agent to **dynamically rebalance a portfolio** across a universe of 11 Indian equities spanning IT and Pharma sectors.

### 7.1 Why RL for Portfolio Management?

Classical optimization (MPT, BL) is **static** — it optimizes once using historical data. Real markets have:
- **Regime changes** (bull → bear, low-vol → high-vol)
- **Non-stationarity** — returns and correlations evolve over time
- **Sequential decision-making** — today's allocation affects tomorrow's starting point

RL addresses this by training a **policy** that maps market observations to portfolio weights, learning through trial-and-error against a risk-adjusted reward signal.

---

### 7.2 The RL Environment

The custom environment `AntiOverfitPortfolioEnv` is built using the **Gymnasium** API (OpenAI Gym successor).

**Ticker Universe (11 stocks):**
- **IT**: TCS, INFY, WIPRO, HCLTECH, TECHM
- **Pharma**: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN, AUROPHARMA

**Date Splits:**

| Split | Period | Rows |
|---|---|---|
| Training | 2005-01-01 → 2020-12-31 | 3,428 days |
| Validation | 2021-01-01 → 2022-12-31 | 496 days |
| Test | 2023-01-01 → present | 823 days |

**Key Environment Parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `STACK_SIZE` | 10 | Days of history the agent sees at once |
| `INITIAL_CAPITAL` | 1.0 | Normalized starting portfolio value |
| `TRANSACTION_COST` | 0.001 | 10 bps per unit of turnover |
| `MAX_WEIGHT` | 0.30 | Max 30% allocation to any single asset |
| `MAX_EP_LEN` | 252 | Episode length = ~1 trading year |
| `random_start` | True (train) | Random start prevents chronological memorization |

---

### 7.3 Feature Engineering (State Space)

11 **stationary** features are engineered per ticker. Raw prices and cumulative returns are deliberately excluded to prevent the agent from memorizing the price path.

| Feature | Formula | Purpose |
|---|---|---|
| `log_ret_1d` | `ln(P_t / P_{t-1})` | 1-day return (reward only, NOT in observation — prevents value leakage) |
| `log_ret_5d` | `ln(P_t / P_{t-5})` | Weekly return momentum |
| `log_ret_20d` | `ln(P_t / P_{t-20})` | Monthly return momentum |
| `vol_10d` | Rolling 10-day std of `log_ret_1d` | Short-term volatility |
| `vol_20d` | Rolling 20-day std of `log_ret_1d` | Medium-term volatility |
| `vol_60d` | Rolling 60-day std of `log_ret_1d` | Long-term volatility |
| `vol_ratio` | `vol_10d / vol_60d` | Volatility regime signal (> 1 = stressed) |
| `rsi_zscore` | RSI(14) → 60-day rolling z-score | Momentum oscillator (stationary) |
| `macd_zscore` | MACD histogram → 60-day rolling z-score | Trend signal (stationary) |
| `ma20_zscore` | `(P/MA20 - mean) / std` over 60 days | Short-term trend deviation |
| `ma50_zscore` | `(P/MA50 - mean) / std` over 60 days | Medium-term trend deviation |

**Observation Vector:**
```
Shape = stack_size × n_assets × n_obs_features + n_assets (current weights)
      = 10 × 11 × 10 + 11
      = 1,111 dimensions

Values: clipped to [-5, 5] (already z-scored in pipeline)
```

---

### 7.4 Action Space

```python
action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32)
```

Raw continuous action → portfolio weights via **softmax with concentration cap**:

```python
scaled  = action * 2.0                         # widen softmax range
exp_a   = exp(scaled - max(scaled))
weights = exp_a / sum(exp_a)                   # standard softmax
weights = clip(weights, 0, MAX_WEIGHT=0.30)    # concentration cap
weights /= sum(weights)                        # re-normalize to 1
```

---

### 7.5 Reward Function (Sortino-style)

The reward is a **risk-adjusted, multi-penalty** signal designed to explicitly discourage the behavior that causes large drawdowns:

```
R = r_net
    - DOWNSIDE_PENALTY  × sigma_down       (penalize asymmetric losses)
    - DRAWDOWN_PENALTY  × drawdown^2       (penalize deep drawdowns heavily)
    - TURNOVER_PENALTY  × turnover         (discourage excessive churn)

R_final = tanh(R × 50)   (squash to (-1, 1) for numerical stability)
```

| Term | Parameter | Value | Purpose |
|---|---|---|---|
| `r_net` | Net step return | `exp(port_log_ret) - 1 - cost` | Actual profit/loss |
| `sigma_down` | Downside std | Std of last 20 negative returns | Asymmetric loss penalization |
| `DOWNSIDE_PENALTY` | λ₁ | 2.0 | Weight on downside risk |
| `drawdown` | Current drawdown | `(peak - value) / peak` | How deep is current hole |
| `DRAWDOWN_PENALTY` | λ₂ | 3.0 | Squared drawdown (heavy for deep holes) |
| `turnover` | Portfolio churn | `Σ|w_new - w_old|` | Total rebalancing magnitude |
| `TURNOVER_PENALTY` | λ₃ | 0.05 | Cost of excessive trading |

**Daily return clipping**: ±15% cap prevents extreme shocks from destabilizing training.
**`tanh` squashing**: Bounds reward to `(-1, 1)`, making training numerically stable.

---

### 7.6 The RecurrentPPO Algorithm (PPO + LSTM)

AssetOS uses **`RecurrentPPO`** from `sb3-contrib`, which extends Proximal Policy Optimization with a **stateful LSTM policy**. This is critical for portfolio management because the agent must remember its own past weights and market history across many steps.

#### Why RecurrentPPO over Standard PPO?

| Standard PPO | RecurrentPPO (PPO + LSTM) |
|---|---|
| Markovian assumption (no memory) | Maintains hidden LSTM state across steps |
| Cannot learn sequential patterns | Can detect regime changes across days |
| Flat MLP policy | MLP + LSTM hybrid policy (`MlpLstmPolicy`) |

#### Algorithm Overview

PPO is an **on-policy** actor-critic algorithm. At each iteration:

1. **Rollout**: Run current policy in the environment for `N_STEPS × N_ENVS` steps, collecting observations, actions, rewards, and values.

2. **Advantage Estimation** via **Generalized Advantage Estimation (GAE)**:
   ```
   A_t = sum( (gamma × lambda)^k × delta_{t+k} )
   delta_t = r_t + gamma × V(s_{t+1}) - V(s_t)
   ```

3. **Policy Update** via clipped surrogate objective:
   ```
   L^CLIP = E[ min( r_t(theta) × A_t,
                    clip(r_t(theta), 1-epsilon, 1+epsilon) × A_t ) ]
   ```

4. **Value Update**: Minimize value function MSE loss against returns.

5. **Entropy Bonus**: Maximize action entropy for exploration:
   ```
   L = L^CLIP - c1 × L^VF + c2 × H[pi_theta]
   ```

#### Network Architecture (`MlpLstmPolicy`)

```
Input (1111-dim observation)
          │
┌─────────▼─────────┐
│   MLP Extractor   │   Processes flat observation
│   [128 → 64]      │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│    LSTM Layer     │   lstm_hidden_size=64, n_lstm_layers=1
│    (64 units)     │   Hidden state carries temporal context
└────┬──────────┬───┘
     │          │
┌────▼───┐  ┌───▼────┐
│Policy  │  │ Value  │   Both heads: MLP [128 → 64]
│ Head   │  │  Head  │
└────┬───┘  └───┬────┘
     │          │
  Action     Value
 (11-dim)   (scalar)
```

Orthogonal weight initialization (`ortho_init=True`) is used for stable early training.

---

### 7.7 Anti-Overfitting Strategy

Preventing overfitting is a primary design concern — training data spans 15 years (2005–2020), and a naive agent could memorize price sequences rather than learn generalizable patterns.

**8 core anti-overfitting mechanisms:**

| Mechanism | Implementation | Effect |
|---|---|---|
| **Random Episode Start** | `reset()` picks random start in training window | Cannot memorize chronological sequence |
| **Stationary Features Only** | No raw prices or cumulative returns in observation | Removes non-stationarity enabling look-ahead |
| **KL-Divergence Early Stop** | `target_kl=0.01` + `EarlyStopOnKL` callback | Halves LR if KL > threshold; prevents destructive updates |
| **Short Episode Length** | `MAX_EP_LEN=252` (1 trading year) | Exposes agent to many distinct market regimes |
| **Small LSTM** | `lstm_hidden_size=64` | Limits memorization capacity |
| **Tight Clip Range** | `CLIP_RANGE=0.1` | Conservative policy updates |
| **Low Epochs** | `N_EPOCHS=4` | Prevents over-fitting on each collected batch |
| **Train/Val Monitoring** | `OverfitMonitor` callback | Warns when val reward drops > 0.1 |
| **VecNormalize** | Running obs & reward normalization | Reduces sensitivity to feature scale differences |

#### Custom Callbacks

**`EarlyStopOnKL`**: Monitors approximate KL divergence during each update. If KL exceeds 0.05, the learning rate is **halved** (rather than stopping entirely) to slow down the policy update.

**`OverfitMonitor`**: Every 25,000 steps, runs `evaluate_policy()` on the validation set and prints mean reward. Warns when validation reward drops more than 0.1 from the previous checkpoint.

---

### 7.8 Training Configuration & Hyperparameters

```python
# Training scale
TOTAL_TIMESTEPS  = 500_000   # Total environment interactions
N_ENVS           = 4         # Parallel environments (DummyVecEnv)

# Optimization
LEARNING_RATE    = 2.5e-4    # Adam optimizer learning rate
N_STEPS          = 256       # Steps per env per update (rollout buffer = 256 × 4 = 1024)
BATCH_SIZE       = 64        # Mini-batch size for gradient updates
N_EPOCHS         = 4         # Optimization epochs per collected rollout

# RL hyperparameters
GAMMA            = 0.995     # Discount factor (long-horizon focus)
GAE_LAMBDA       = 0.95      # GAE smoothing parameter
CLIP_RANGE       = 0.1       # PPO clipping (conservative)
ENT_COEF         = 0.05      # Entropy coefficient (strong exploration pressure)
TARGET_KL        = 0.01      # Built-in early stop per update epoch
VF_COEF          = 0.5       # Value function loss coefficient
MAX_GRAD_NORM    = 0.5       # Gradient clipping

# Architecture
LSTM_HIDDEN      = 64        # LSTM hidden state size
N_LSTM_LAYERS    = 1         # Number of LSTM layers
POLICY_NET_ARCH  = [128, 64] # Policy head (pi)
VALUE_NET_ARCH   = [128, 64] # Value head (vf)
SEED             = 42        # Reproducibility

# Reward shaping
DOWNSIDE_PENALTY = 2.0       # Multiplier on downside deviation
DRAWDOWN_PENALTY = 3.0       # Multiplier on squared drawdown
TURNOVER_PENALTY = 0.05      # Multiplier on portfolio turnover
TRANSACTION_COST = 0.001     # 10 bps per unit of turnover (actual cost)
```

---

### 7.9 RL-Driven Rebalancing in the Dashboard

The `/api/rl_rebalance` endpoint implements a **rule-based RL proxy** (heuristic approximation of trained PPO behavior) that runs in real-time on the current portfolio:

1. **Market Regime Detection** (rolling volatility heuristic):
   - `avg_vol > 35%` → **Risk-Off** (reduce equity by 8%)
   - `avg_vol > 25%` → **Volatile** (reduce equity by 3%)
   - `avg_vol < 15%` → **Bullish** (increase equity by 5%)
   - Otherwise → **Normal**

2. **Risk Parity Rebalancing**:
   ```
   MRC_i = (Sigma × w)_i / sigma_p
   RC_i  = w_i × MRC_i
   Delta_w = -(RC - mean(RC)) × 0.3    [damped adjustment toward equal risk contribution]
   ```

3. **Explainability Layer**: Natural-language reasons for each rebalancing decision are returned (regime type, risk concentration shifts, Sharpe improvement).

4. **PPO Training Curve Simulation**: Generates a 60-epoch reward/policy-loss/entropy curve for the `RLRebalancingEngine.jsx` visualization.

---

## 8. Institutional Stress Testing Engine

`stress_testing.py` implements a **Goldman Sachs / BlackRock-grade** stress testing framework across 4 analytical layers (1,513 lines).

### Layer 1: Macro-Economic Scenario Stress Testing

8 predefined historical and hypothetical crisis scenarios:

| Scenario | IT Shock | Banking Shock | Energy Shock | Pharma Shock | Vol Multiplier | Correlation Override |
|---|---|---|---|---|---|---|
| Global Financial Crisis (2008) | -42% | -55% | -45% | -20% | 3.0× | 0.90 |
| COVID-19 Crash (2020) | -25% | -38% | -40% | +10% | 2.5× | 0.75 |
| Interest Rate Shock | -15% | -25% | -10% | -8% | 1.8× | 0.55 |
| Stagflation / Inflation Shock | -25% | -18% | +15% | -10% | 2.0× | 0.45 |
| Liquidity Crisis | -30% | -50% | -35% | -15% | 3.5× | 0.85 |
| Sector Collapse — IT/Tech | -60% | -10% | -5% | +2% | 2.0× | 0.30 |
| Sector Collapse — Banking | -12% | -55% | -15% | -5% | 2.2× | 0.50 |
| Bull Market Overextension | -28% | -22% | -18% | -12% | 1.5× | 0.65 |

Each scenario applies sector-level shocks to portfolio tickers via a `SECTOR_MAP` covering 40+ Nifty 100 stocks. A `recovery_factor` models partial recovery within the analysis window.

### Layer 2: Extreme Value / Tail Risk Analysis (EVT)

- Fits **Generalized Pareto Distribution (GPD)** to the worst 10% of historical returns
- Computes VaR and CVaR at 95%, 99%, 99.5% confidence levels
- Uses **Student-t distribution** from `scipy.stats` for realistic heavy-tail modeling

### Layer 3: Sensitivity & Stability Analysis

- **Correlation stress**: Tests portfolio under correlation spikes (e.g., all correlations → 0.90 in crisis)
- **Volatility sensitivity**: Applies scenario-specific `volatility_multiplier` to covariance matrix
- **Recovery modeling**: Partial recovery within analysis window via `recovery_factor`

### Layer 4: IPS Constraint Validation Under Stress

For each scenario, checks whether the stressed portfolio still satisfies IPS constraints:
- Equity allocation within `EquityMin / EquityMax`
- Max drawdown within `MaxDrawdown`
- Returns violation details and flags each constraint breach

### Sector Classification Map (40+ Nifty 100 Stocks)

```python
SECTOR_MAP = {
    "IT":      ["INFY", "TCS", "HCLTECH", "WIPRO", "TECHM"],
    "Pharma":  ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "BIOCON", "AUROPHARMA"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK"],
    "Energy":  ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC"],
    "FMCG":    ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "Auto":    ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ_AUTO", "HEROMOTOCO"],
    "Metals":  ["TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA"],
}
```

---

## 9. Backend API — Flask REST Server

`backend/app.py` — 368 lines, 7 endpoints. CORS-enabled for React frontend communication.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check |
| `POST` | `/api/onboard` | Submit investor profile, receive IPS constraints |
| `POST` | `/api/generate_portfolio` | Run MPT, return allocation + full Monte Carlo risk data |
| `POST` | `/api/optimize_bl` | Run Black-Litterman with user views |
| `POST` | `/api/resimulate` | Re-run Monte Carlo with same weights (fresh randomness) |
| `GET` | `/api/assets` | List all available ticker symbols from data directory |
| `POST` | `/api/rl_rebalance` | RL-driven regime-aware portfolio rebalancing |

### Global State

```python
engine = PortfolioBuilder()   # Shared instance (loaded once at startup)
_last_weights = None          # Cached for re-simulation and RL rebalancing
```

---

## 10. Frontend — React Web Interface

Built with **React 19 + Vite 7 + TailwindCSS v4**, the frontend follows an "Old Money" / institutional aesthetic with gold accents on a dark navy palette.

### Design System (CSS Custom Properties)

```css
--elite-navy:     #0a192f    /* Page background */
--elite-charcoal: #112240    /* Card background */
--elite-gold:     #d4af37    /* Primary accent */
--elite-cream:    #e6f1ff    /* Primary text */
--elite-slate:    #8892b0    /* Secondary text */
```

### Pages / Components

| Component | File | Purpose |
|---|---|---|
| **Onboarding** | `Onboarding.jsx` (14KB) | Multi-step investor intake form with animated transitions |
| **Dashboard** | `Dashboard.jsx` (27KB) | Main portfolio view: allocation pie, Monte Carlo cone, scorecard, benchmarks |
| **MarketViews** | `MarketViews.jsx` (8KB) | Stock view input form for Black-Litterman re-optimization |
| **RLRebalancingEngine** | `RLRebalancingEngine.jsx` (22KB) | RL rebalancing: regime detection, allocation transition, training curve visualization |

### Frontend Features

**Onboarding**: Animated multi-step form collecting 10 investor parameters.

**Dashboard**:
- Recharts `PieChart` for portfolio allocation breakdown
- Recharts `AreaChart` for Monte Carlo cone (P5/P25/P50/P75/P95 bands)
- Live portfolio metrics (Sharpe, Sortino, VaR, CVaR, Max Drawdown)
- Portfolio Scorecard with A+–D letter grades across 5 dimensions
- Benchmark comparison table (vs. Nifty 50, FD, Savings)
- Goal probability gauge (% chance of achieving financial goal)

**RLRebalancingEngine**:
- Animated `AllocationRing` (rotating SVG pie) showing regime-aware allocations
- Allocation transition bar chart (before vs. after rebalancing per ticker)
- PPO training reward/policy-loss/entropy curves (Recharts LineChart)
- Explainability cards with natural language + confidence scores
- Live portfolio value ticker (30-day simulated daily performance)

### Vite Proxy Configuration

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': 'http://localhost:5000'
  }
}
```

---

## 11. Data Layer

### Primary Data Source: `NIFTY100_Sectors_Initial/`

Individual stock CSVs for Nifty 100 stocks, organized by sector. Each file must contain:

```
Date,Adj Close
2020-01-01,1234.50
2020-01-02,1250.00
```

The `load_market_data()` method pipeline:
1. Globs all `*.csv` in the data directory
2. Reads each file, preferring `Adj Close` over `Close`
3. Converts to numeric, coercing string header rows to NaN
4. **Forward-fills** missing values, then drops remaining NaN rows
5. Computes log returns: `log(P_t / P_{t-1})`
6. Annualizes: `mu = mean(returns) × 252`, `Sigma = cov(returns) × 252`
7. **Caches** result in `_market_data_cache` for O(1) subsequent calls

### Investor Profile Data: `AssetOS_Data/investor_profile.csv`

| Column | Type | Values |
|---|---|---|
| `InvestorID` | int | Unique identifier |
| `RiskTolerance` | str | `conservative`, `moderate`, `aggressive` |
| `InvestmentHorizon` | float | Years (e.g., 5, 10, 20, 30) |
| `LiquidityNeed` | str | `low`, `medium`, `high` |

### Output Files

| File | Description |
|---|---|
| `Portfolio_<ClientName>.csv` | Final portfolio weights (from CLI run) |
| `portfolio_allocation.csv` | Latest allocation output |
| `asset_os.log` | All runtime events |
| `portfolio_performance.png` | RL agent portfolio value chart |
| `validation_set.png` | Agent validation performance |
| `test_set.png` | Agent test performance |

---

## 12. Data Diagnostic Tool

`Diagnostic.py` runs two critical checks before optimization:

### Check 1: Risk Premium Audit

Verifies each stock has a positive risk premium (expected return > risk-free rate of 6.5%):

```
TICKER          RETURN (Ann)    STATUS
────────────────────────────────────────
TCS             14.23%          ✅ Healthy
RELIANCE        12.87%          ✅ Healthy
WIPRO            5.91%          ⚠️  LOW RETURN
```

Assets below the risk-free rate get flagged — the SLSQP optimizer will naturally push them to zero weight, potentially distorting the efficient frontier.

### Check 2: Correlation Audit

Computes average portfolio correlation:
- `avg_corr > 0.5` → **High Correlation Warning**: diversification power is weak, efficient frontier degenerates.
- `avg_corr ≤ 0.5` → Acceptable diversification.

---

## 13. Installation & Setup

### Prerequisites

- **Python 3.9+** (tested on 3.12.x)
- **Node.js 18+** and **npm**
- macOS / Linux

### Step 1: Navigate to Project

```bash
cd /path/to/Major_Project
```

### Step 2: Install Python Dependencies

```bash
pip install flask flask-cors numpy pandas scipy matplotlib seaborn
pip install yfinance gymnasium stable-baselines3 sb3-contrib pytz
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4: Verify Data Health

```bash
python Diagnostic.py
```

---

## 14. Run Commands

### Option 1 — One-Click Launcher (Recommended)

```bash
python run_gui.py
```

> Opens at **http://localhost:5173** after ~5 seconds. Press `Ctrl+C` to shut down both services.

### Option 2 — Run Services Manually

**Terminal 1 — Backend (Flask API, Port 5000):**
```bash
python backend/app.py
```

**Terminal 2 — Frontend (React/Vite, Port 5173):**
```bash
cd frontend
npm run dev
```

### Option 3 — Command-Line Mode (No Web UI)

```bash
python IPS.py
```

Steps: Onboarding → MPT → (Optional) Black-Litterman → Monte Carlo → Save CSV

### Option 4 — Train the RL Agent

```bash
jupyter notebook PPO_training.ipynb
```

### Option 5 — Data Diagnostics

```bash
python Diagnostic.py
```

### Other Utility Commands

```bash
# Test Black-Litterman in isolation
python test_bl.py

# Generate investor constraints from profile CSV
python investor_profiler.py

# Run system health validator
python system_validator.py

# Build frontend for production
cd frontend && npm run build

# Preview production build
cd frontend && npm run preview

# View RL training progress
tensorboard --logdir drl_tensorboard/
```

---

## 15. API Reference

All endpoints on **`http://localhost:5000`**.

### POST `/api/onboard`

**Request Body:**
```json
{
  "name": "Rahul Sharma",
  "goal": "Retirement",
  "capital": 500000,
  "liquidity": 20,
  "risk_attitude": 3,
  "goal_years": 15,
  "age": 35,
  "loss_tolerance": 20
}
```

**Response:**
```json
{
  "message": "Profile created",
  "profile": { ... },
  "ips": {
    "ReturnObjective": 0.118,
    "EquityMin": 0.50,
    "EquityMax": 0.70,
    "CashMin": 0.05,
    "CashMax": 0.15,
    "MaxDrawdown": 0.20
  }
}
```

### POST `/api/generate_portfolio`

**Response includes:**
- `allocation` — array of `{name, value}` (% weights)
- `risk_analysis` — full Monte Carlo output including:
  - `months`, `p5`, `p25`, `median`, `p75`, `p95` — cone data
  - `sample_paths` — 6 individual GBM paths
  - `portfolio_metrics` — Sharpe, Sortino, VaR, CVaR, Max DD, Diversification Ratio
  - `scorecard` — letter grades for 5 dimensions
  - `benchmarks` — vs. Nifty 50, FD, Savings
  - `goal_probability` — % of paths achieving target wealth
  - `final_stats` — worst/median/best case terminal wealth

### POST `/api/optimize_bl`

**Request:**
```json
{
  "views": {
    "RELIANCE": 0.25,
    "INFY": 0.15,
    "TCS": -0.05
  }
}
```

**Response:** Same structure as `/api/generate_portfolio` with BL-adjusted allocation.

### POST `/api/rl_rebalance`

**Response includes:**
- `regime` — `{name, description, avg_volatility}`
- `metrics` — `{sharpe, sortino, max_drawdown, win_rate, volatility, old_sharpe, ...}`
- `allocation_transition` — `[{name, prev, curr, diff}]`
- `reward_curve` — 60-epoch `[{epoch, reward, policyLoss, entropyLoss}]`
- `explain_reasons` — `[{text, icon, confidence}]`
- `live_data` — 30-day simulated portfolio value curve
- `pie_data` — new allocation for pie chart

---

## 16. Key Configuration

| Parameter | Location | Value | Notes |
|---|---|---|---|
| Risk-Free Rate | `IPS.py:CONFIG` | 6.5% | India 10Y gilt proxy |
| Trading Days | `IPS.py:CONFIG` | 252 | NSE calendar |
| Data Directory | `IPS.py:CONFIG` | `NIFTY100_Sectors_Initial` | Primary price CSVs |
| Monte Carlo Paths | `IPS.py` | 1,000 | `run_full_simulation()` |
| Simulation Horizon | `IPS.py` | 1–30 years | From user's `goal_years` |
| BL Delta (δ) | `IPS.py` | 2.5 | Risk aversion coefficient |
| BL Tau (τ) | `IPS.py` | 0.05 | Equilibrium uncertainty |
| Flask Port | `backend/app.py` | 5000 | |
| Vite Port | `frontend/vite.config.js` | 5173 | |
| Max Asset Weight (RL) | `PPO_training.ipynb` | 30% | Concentration cap |
| RL Training Steps | `PPO_training.ipynb` | 500,000 | Total environment steps |
| RL Episode Length | `PPO_training.ipynb` | 252 | ~1 trading year |

---

## 17. Logging

All runtime events are written to `asset_os.log` in the project root:

```
2026-03-15 10:23:45 [INFO] Onboarding complete for: Rahul Sharma
2026-03-15 10:23:48 [INFO] Optimizer constraints: {'ReturnObjective': 0.118, ...}
2026-03-15 10:23:51 [WARNING] Optimization failed to converge. Returning initial guess.
2026-03-15 10:23:52 [ERROR] Matrix inversion failed in BL model: singular matrix
```

Log format: `%(asctime)s [%(levelname)s] %(message)s`

---

## 18. Saved Model Artifacts

The project includes several pre-trained PPO model checkpoints representing different training runs:

| File | Size | Description |
|---|---|---|
| `ppo_antioverfitting.zip` | ~7.4 MB | **Final model** — all anti-overfitting mechanisms active |
| `vecnorm_antioverfitting.pkl` | ~47 KB | Running normalization stats (required for inference with anti-overfit model) |
| `ppo_fixed.zip` | ~1.5 MB | Earlier fixed-bug variant |
| `vecnorm_fixed.pkl` | ~7 KB | Normalization stats for fixed model |
| `ppo_lstm_portfolio.zip` | ~3.0 MB | LSTM variant (intermediate experiment) |
| `ppo_portfolio_nse.zip` | ~4.1 MB | Trained on broader NSE universe |
| `vecnorm_nse.pkl` | ~8 KB | Normalization stats for NSE model |
| `ppo_portfolio_rebalancer.zip` | ~747 KB | Lightweight rebalancer variant |

**To load and run inference:**

```python
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

model = RecurrentPPO.load("ppo_antioverfitting")
vec_env = VecNormalize.load("vecnorm_antioverfitting.pkl",
                            DummyVecEnv([lambda: your_env]))
vec_env.training    = False   # Freeze normalization stats for inference
vec_env.norm_reward = False
```

**View TensorBoard training logs:**
```bash
tensorboard --logdir drl_tensorboard/
tensorboard --logdir drl_tensorboard_lstm/
```

---

*Built as a Major Project demonstrating the intersection of quantitative finance, deep reinforcement learning, and modern full-stack web development focused on the Indian equity market.*
