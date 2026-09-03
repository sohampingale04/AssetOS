"""
ASSETOS Phase 5 – PortfolioEnv
Custom Gymnasium environment for DRL-based portfolio rebalancing.

State  : returns history + vol + momentum + current weights + portfolio metrics
Action : continuous portfolio weight vector (softmax projected)
Reward : risk-adjusted = return − 0.5*vol − 0.1*turnover − 0.3*drawdown
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Helper ────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def project_simplex(v: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the unit simplex with an optional per-asset cap.
    Uses iterative clipping so that sum(w) = 1 and 0 ≤ w ≤ max_weight.
    """
    n    = len(v)
    # First: softmax to get a probability vector
    w    = softmax(v)
    # Then: clip to max_weight and re-normalise (iterate to convergence)
    for _ in range(50):
        w = np.clip(w, 0.0, max_weight)
        s = w.sum()
        if abs(s - 1.0) < 1e-9:
            break
        w = w / s
    return w.astype(np.float32)


# ── Environment ───────────────────────────────────────────────────────────────

class PortfolioEnv(gym.Env):
    """
    Parameters
    ----------
    feature_array : np.ndarray shape (T, N, F)
        Pre-normalised feature matrix.
    asset_returns : np.ndarray shape (T, N)
        Simple period returns (aligned with feature_array).
    tickers : list[str]
        Asset names (for logging).
    lookback : int
        Number of past periods used to build the return-history part of state.
    initial_capital : float
    transaction_cost : float
        Fractional one-way cost per unit of turnover.
    max_weight : float
        IPS per-asset cap.
    reward_weights : dict
        Coefficients for each reward component.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_array: np.ndarray,
        asset_returns: np.ndarray,
        tickers: list[str],
        lookback: int = 20,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.001,
        max_weight: float = 0.20,
        reward_weights: dict | None = None,
    ):
        super().__init__()

        assert feature_array.shape[0] == asset_returns.shape[0], \
            "feature_array and asset_returns must have same time dimension."

        self.features     = feature_array.astype(np.float32)   # (T, N, F)
        self.returns      = asset_returns.astype(np.float32)    # (T, N)
        self.tickers      = tickers
        self.N            = len(tickers)              # number of assets
        self.F            = feature_array.shape[2]    # features per asset
        self.T            = feature_array.shape[0]
        self.lookback     = lookback
        self.init_capital = initial_capital
        self.tc           = transaction_cost
        self.max_weight   = max_weight

        self.rw = reward_weights or {
            "return":   1.0,
            "vol":     -0.5,
            "turnover":-0.1,
            "drawdown":-0.3,
        }

        # ── Action Space ──────────────────────────────────────────────────────
        # Raw logits; projected onto simplex in step()
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N,), dtype=np.float32
        )

        # ── Observation Space ─────────────────────────────────────────────────
        # Components:
        #   [0] return history       : lookback × N
        #   [1] current features     : N × F
        #   [2] current weights      : N
        #   [3] portfolio scalars    : 4  (ret, vol, sharpe, drawdown)
        obs_dim = (self.lookback * self.N) + (self.N * self.F) + self.N + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _reset_state(self):
        self.t            = self.lookback          # start after warm-up
        self.weights      = np.ones(self.N, dtype=np.float32) / self.N
        self.portfolio_val = self.init_capital
        self.peak         = self.init_capital
        self.return_hist: list[float] = []
        self.val_hist: list[float]    = [self.init_capital]

    def _portfolio_scalars(self) -> np.ndarray:
        """Return [recent_ret, recent_vol, approx_sharpe, drawdown]."""
        if len(self.return_hist) < 2:
            return np.zeros(4, dtype=np.float32)

        hist  = np.array(self.return_hist[-52:])   # ~1 year of weekly data
        ret   = float(hist.mean())
        vol   = float(hist.std() + 1e-8)
        sharpe = ret / vol
        dd    = float((self.portfolio_val - self.peak) / (self.peak + 1e-8))
        return np.array([ret, vol, sharpe, dd], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        # Return history block: lookback × N
        start = max(0, self.t - self.lookback)
        ret_block = self.returns[start:self.t]             # (lookback, N)
        if len(ret_block) < self.lookback:
            pad       = np.zeros((self.lookback - len(ret_block), self.N), dtype=np.float32)
            ret_block = np.vstack([pad, ret_block])

        feat_block   = self.features[self.t]               # (N, F)
        scalars      = self._portfolio_scalars()           # (4,)

        obs = np.concatenate([
            ret_block.flatten(),          # lookback * N
            feat_block.flatten(),         # N * F
            self.weights,                 # N
            scalars,                      # 4
        ])
        return obs.astype(np.float32)

    # ── Gym Interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        # 1. Project action onto constrained simplex
        new_weights = project_simplex(action, self.max_weight)

        # 2. Turnover
        turnover = float(np.abs(new_weights - self.weights).sum())

        # 3. Transaction cost
        cost = turnover * self.tc

        # 4. Portfolio return for this period
        period_ret_vec = self.returns[self.t]              # (N,)
        gross_ret      = float(np.dot(self.weights, period_ret_vec))
        net_ret        = gross_ret - cost

        # 5. Update portfolio value
        self.portfolio_val *= (1.0 + net_ret)
        self.peak          = max(self.peak, self.portfolio_val)

        # 6. Track history
        self.return_hist.append(net_ret)
        self.val_hist.append(self.portfolio_val)

        # 7. Reward components
        recent_rets = np.array(self.return_hist[-20:])
        r_vol       = float(recent_rets.std() + 1e-8) if len(recent_rets) > 1 else 0.0
        r_dd        = float((self.portfolio_val - self.peak) / (self.peak + 1e-8))

        reward = (
            self.rw["return"]   * net_ret
            + self.rw["vol"]    * r_vol
            + self.rw["turnover"] * turnover
            + self.rw["drawdown"] * abs(r_dd)
        )

        # 8. Update weights and timestep
        self.weights = new_weights
        self.t      += 1

        done      = self.t >= self.T - 1
        truncated = False

        info = {
            "portfolio_value": self.portfolio_val,
            "net_return":      net_ret,
            "turnover":        turnover,
            "drawdown":        r_dd,
            "weights":         self.weights.copy(),
        }

        return self._build_obs(), float(reward), done, truncated, info

    def render(self):
        print(
            f"  t={self.t:4d} | val={self.portfolio_val:.4f} | "
            f"ret={self.return_hist[-1] if self.return_hist else 0:.4f} | "
            f"weights={np.round(self.weights, 3)}"
        )

    def action_masks(self) -> np.ndarray:
        """Returns ones (all actions valid) – provided for compatibility."""
        return np.ones(self.N, dtype=bool)


if __name__ == "__main__":
    # Quick smoke test with random data
    T, N, F = 500, 5, 8
    feats   = np.random.randn(T, N, F).astype(np.float32)
    rets    = np.random.randn(T, N).astype(np.float32) * 0.01
    tickers = [f"STK{i}" for i in range(N)]

    env = PortfolioEnv(feats, rets, tickers)
    obs, _ = env.reset()
    print("Obs dim:", obs.shape)

    for _ in range(10):
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        env.render()
        if done:
            break
    print("Smoke test passed.")
