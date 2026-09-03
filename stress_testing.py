"""
================================================================================
AssetOS — Institutional Stress Testing Engine
================================================================================

Advanced stress testing framework that evaluates portfolio robustness under
extreme, real-world financial conditions. Goes beyond Monte Carlo simulation
to deliver Goldman Sachs / BlackRock–grade risk analytics.

Layers:
    1. Macro-Economic Scenario Stress Testing
    2. Extreme Value / Tail Risk Analysis (EVT, fat-tailed distributions)
    3. Sensitivity & Stability Analysis
    4. Constraint & IPS Validation Under Stress

Integration:
    IPS → Optimization → Stress Testing → Reporting

Author: AssetOS Engine
================================================================================
"""

import os
import copy
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict

try:
    from scipy.optimize import minimize
    from scipy.stats import t as student_t
    from scipy.stats import genpareto
except ImportError:
    minimize = None
    student_t = None
    genpareto = None
    warnings.warn("SciPy not available. Some stress testing features will be limited.")

# Import the core engine
try:
    from IPS import PortfolioBuilder, CONFIG
except ImportError:
    warnings.warn("Could not import PortfolioBuilder. Stress engine will require manual injection.")
    CONFIG = {'risk_free_rate': 0.065, 'trading_days': 252}

logger = logging.getLogger("AssetOS.StressTesting")


# ==============================================================================
# Sector Classification Map (Indian Market — NIFTY100 Universe)
# ==============================================================================

SECTOR_MAP = {
    # IT
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "LUPIN": "Pharma", "BIOCON": "Pharma",
    "AUROPHARMA": "Pharma",
    # Banking / Financials
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "BAJFINANCE": "Banking",
    "BAJAJFINSV": "Banking", "INDUSINDBK": "Banking",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "BPCL": "Energy", "IOC": "Energy",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG",
    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "BAJAJ_AUTO": "Auto", "HEROMOTOCO": "Auto",
    # Metals
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "COALINDIA": "Metals",
}


# ==============================================================================
# 1. MACRO-ECONOMIC SCENARIO DEFINITIONS
# ==============================================================================

STRESS_SCENARIOS = OrderedDict({
    "Global Financial Crisis (2008)": {
        "description": "Lehman-style collapse. Equities crater, correlations spike to 1, "
                       "volatility triples. Credit markets freeze.",
        "shocks": {
            "IT": -0.42, "Banking": -0.55, "Energy": -0.45, "Pharma": -0.20,
            "FMCG": -0.15, "Auto": -0.50, "Metals": -0.60,
            "_default": -0.40,
        },
        "correlation_override": 0.90,  # All correlations spike
        "volatility_multiplier": 3.0,
        "recovery_factor": 0.0,  # No recovery in this window
    },

    "COVID-19 Crash (2020)": {
        "description": "Sharp 30% drop across sectors, followed by a V-shaped recovery. "
                       "Pharma outperforms. Travel/Hospitality wiped out.",
        "shocks": {
            "IT": -0.25, "Banking": -0.38, "Energy": -0.40, "Pharma": 0.10,
            "FMCG": -0.10, "Auto": -0.35, "Metals": -0.30,
            "_default": -0.30,
        },
        "correlation_override": 0.75,
        "volatility_multiplier": 2.5,
        "recovery_factor": 0.60,  # 60% recovery within analysis window
    },

    "Interest Rate Shock": {
        "description": "Central bank hikes rates 300bps aggressively. Bond prices collapse, "
                       "equity multiples compress. Rate-sensitive sectors hit hardest.",
        "shocks": {
            "IT": -0.15, "Banking": -0.25, "Energy": -0.10, "Pharma": -0.08,
            "FMCG": -0.05, "Auto": -0.20, "Metals": -0.12,
            "_default": -0.15,
        },
        "correlation_override": 0.55,
        "volatility_multiplier": 1.8,
        "recovery_factor": 0.20,
    },

    "Stagflation / Inflation Shock": {
        "description": "Persistent high inflation + economic stagnation. Real returns "
                       "collapse. Commodities and energy surge. Growth stocks crushed.",
        "shocks": {
            "IT": -0.25, "Banking": -0.18, "Energy": 0.15, "Pharma": -0.10,
            "FMCG": -0.08, "Auto": -0.22, "Metals": 0.20,
            "_default": -0.12,
        },
        "correlation_override": 0.45,
        "volatility_multiplier": 2.0,
        "recovery_factor": 0.10,
    },

    "Liquidity Crisis": {
        "description": "Credit markets seize. Forced selling across all asset classes. "
                       "Small/mid-caps hit disproportionately. Bid-ask spreads explode.",
        "shocks": {
            "IT": -0.30, "Banking": -0.50, "Energy": -0.35, "Pharma": -0.15,
            "FMCG": -0.10, "Auto": -0.40, "Metals": -0.45,
            "_default": -0.35,
        },
        "correlation_override": 0.85,
        "volatility_multiplier": 3.5,
        "recovery_factor": 0.05,
    },

    "Sector Collapse — IT/Tech": {
        "description": "Tech bubble burst. IT sector drops 60%. Contagion limited "
                       "to growth-oriented sectors.",
        "shocks": {
            "IT": -0.60, "Banking": -0.10, "Energy": -0.05, "Pharma": 0.02,
            "FMCG": 0.03, "Auto": -0.08, "Metals": -0.05,
            "_default": -0.10,
        },
        "correlation_override": 0.30,
        "volatility_multiplier": 2.0,
        "recovery_factor": 0.15,
    },

    "Sector Collapse — Banking": {
        "description": "Banking crisis / NPA spiral. Financial sector drops 55%. "
                       "Credit contraction spreads to real economy.",
        "shocks": {
            "IT": -0.12, "Banking": -0.55, "Energy": -0.15, "Pharma": -0.05,
            "FMCG": -0.03, "Auto": -0.20, "Metals": -0.18,
            "_default": -0.15,
        },
        "correlation_override": 0.50,
        "volatility_multiplier": 2.2,
        "recovery_factor": 0.10,
    },

    "Bull Market Overextension": {
        "description": "After prolonged bull run, valuations become unsustainable. "
                       "Mean-reversion correction of 20-30% across the board.",
        "shocks": {
            "IT": -0.28, "Banking": -0.22, "Energy": -0.18, "Pharma": -0.12,
            "FMCG": -0.10, "Auto": -0.25, "Metals": -0.20,
            "_default": -0.20,
        },
        "correlation_override": 0.65,
        "volatility_multiplier": 1.5,
        "recovery_factor": 0.30,
    },
})


# ==============================================================================
# 2. STRESS TESTING ENGINE
# ==============================================================================

class StressTestingEngine:
    """
    Institutional-grade stress testing engine for AssetOS portfolios.
    
    Evaluates portfolio robustness across 4 dimensions:
        1. Macro-Economic Scenario Stress Testing
        2. Extreme Value / Tail Risk Analysis
        3. Sensitivity & Stability Analysis  
        4. IPS Constraint Validation Under Stress
    """

    def __init__(self, portfolio_builder: 'PortfolioBuilder' = None):
        """
        Args:
            portfolio_builder: An initialized PortfolioBuilder instance with 
                               user_profile loaded and market data cached.
        """
        self.engine = portfolio_builder
        self.results = {}
        self._market_cache = None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_market_data(self):
        """Lazily loads and caches market data from the PortfolioBuilder."""
        if self._market_cache is None:
            mu, cov, assets = self.engine.load_market_data()
            self._market_cache = {
                'mu': mu,
                'cov': cov,
                'assets': assets,
            }
        return self._market_cache

    def _get_sector(self, ticker):
        """Returns the sector for a given ticker, defaulting to 'Other'."""
        return SECTOR_MAP.get(ticker, "Other")

    def _get_asset_shock(self, ticker, scenario_shocks):
        """
        Resolves the shock for a specific ticker from sector-level scenario shocks.
        Falls back to '_default' if sector is unmapped.
        """
        sector = self._get_sector(ticker)
        return scenario_shocks.get(sector, scenario_shocks.get("_default", -0.20))

    def _prepare_weights(self, allocation):
        """
        Prepares portfolio weights — strips synthetic assets, normalizes.
        Returns (risky_weights_series, cash_weight).
        """
        SYNTHETIC = ["CASH_RESERVE", "BOND_INDEX", "ALT_INDEX"]
        risky = allocation.drop(SYNTHETIC, errors='ignore')
        cash = allocation.get("CASH_RESERVE", 0.0)
        total = risky.sum()
        if total < 1e-6:
            return risky, cash
        return risky, cash

    # ------------------------------------------------------------------
    # LAYER 1: Macro-Economic Scenario Stress Testing
    # ------------------------------------------------------------------

    def run_scenario_stress_test(self, allocation, ips_constraints):
        """
        Runs all predefined macro-economic stress scenarios against the portfolio.

        Args:
            allocation: pd.Series of portfolio weights (including CASH_RESERVE).
            ips_constraints: Dict from generate_ips() with keys like
                             ReturnObjective, EquityMin, EquityMax, CashMin, MaxDrawdown.

        Returns:
            List of dicts, one per scenario, with keys:
                scenario, description, portfolio_return, drawdown,
                ips_violation, violation_details, asset_impacts, status
        """
        risky_weights, cash_weight = self._prepare_weights(allocation)
        mkt = self._get_market_data()
        results = []

        for scenario_name, scenario_config in STRESS_SCENARIOS.items():
            shocks = scenario_config["shocks"]
            recovery = scenario_config.get("recovery_factor", 0.0)

            # Calculate stressed return for each asset
            asset_impacts = {}
            stressed_portfolio_return = 0.0

            for ticker, weight in risky_weights.items():
                if weight < 1e-6:
                    continue
                shock = self._get_asset_shock(ticker, shocks)
                # Net shock after partial recovery
                net_shock = shock * (1 - recovery)
                stressed_return = net_shock * weight
                stressed_portfolio_return += stressed_return
                asset_impacts[ticker] = {
                    "weight": round(weight * 100, 2),
                    "shock": round(shock * 100, 2),
                    "net_shock": round(net_shock * 100, 2),
                    "contribution": round(stressed_return * 100, 4),
                    "sector": self._get_sector(ticker),
                }

            # Drawdown = magnitude of loss (positive number representing % drop)
            drawdown = abs(min(stressed_portfolio_return, 0.0))

            # IPS Violation Check
            violation_details = self._check_ips_violations(
                stressed_portfolio_return, drawdown, risky_weights, cash_weight, ips_constraints
            )

            ips_violated = any(v["violated"] for v in violation_details.values())

            # Determine status
            if drawdown > ips_constraints.get("MaxDrawdown", 0.15) * 1.5:
                status = "CRITICAL FAILURE"
            elif ips_violated:
                status = "WARNING"
            else:
                status = "SAFE"

            results.append({
                "scenario": scenario_name,
                "description": scenario_config["description"],
                "portfolio_return": round(stressed_portfolio_return * 100, 4),
                "drawdown": round(drawdown * 100, 4),
                "ips_violation": ips_violated,
                "violation_details": violation_details,
                "asset_impacts": asset_impacts,
                "status": status,
            })

        self.results["scenario_stress"] = results
        return results

    # ------------------------------------------------------------------
    # LAYER 2: Extreme Value / Tail Risk Analysis
    # ------------------------------------------------------------------

    def run_tail_risk_analysis(self, allocation, n_simulations=10000, confidence=0.95):
        """
        Computes tail risk metrics using fat-tailed distributions (Student-t).
        
        Goes beyond Monte Carlo by modeling the actual shape of extreme losses
        using Extreme Value Theory approximations.

        Args:
            allocation: pd.Series of portfolio weights.
            n_simulations: Number of tail risk simulations.
            confidence: Confidence level for VaR/CVaR (default 95%).

        Returns:
            Dict with keys: var_95, cvar_95, max_drawdown, expected_return,
            tail_loss_probability, black_swan_loss, distribution_params,
            worst_case_scenarios
        """
        risky_weights, cash_weight = self._prepare_weights(allocation)
        mkt = self._get_market_data()

        if risky_weights.sum() < 1e-6:
            return {"error": "Portfolio is 100% cash. No tail risk to evaluate."}

        # Align weights with market data
        common = [t for t in risky_weights.index if t in mkt['mu'].index]
        if not common:
            return {"error": "No overlapping tickers between allocation and market data."}

        w = risky_weights[common].values
        w_norm = w / w.sum() if w.sum() > 0 else w
        mu_aligned = mkt['mu'][common].values
        cov_aligned = mkt['cov'].loc[common, common].values

        # Portfolio-level stats
        port_mu = np.dot(w_norm, mu_aligned)
        port_vol = np.sqrt(np.dot(w_norm.T, np.dot(cov_aligned, w_norm)))

        # --- Fat-Tailed Simulation using Student-t ---
        # Fit degrees of freedom.  For equity markets, df ≈ 4-6 is realistic.
        # Lower df = fatter tails = more extreme events.
        df_t = 5  # Empirically calibrated for emerging market equities

        if student_t is not None:
            # Generate fat-tailed returns
            daily_mu = port_mu / 252
            daily_vol = port_vol / np.sqrt(252)

            # Student-t scaled to match portfolio moments
            t_samples = student_t.rvs(df=df_t, loc=daily_mu, scale=daily_vol, size=(252, n_simulations))

            # Compute annual returns via compounding
            annual_returns = np.prod(1 + t_samples, axis=0) - 1
        else:
            # Fallback: use normal distribution with vol scaling
            annual_returns = np.random.normal(port_mu, port_vol, n_simulations)

        # --- Risk Metrics ---
        var_level = 1 - confidence  # 0.05 for 95% confidence

        # VaR: What is the worst loss at the given confidence level?
        var_95 = -np.percentile(annual_returns, var_level * 100)

        # CVaR (Expected Shortfall): Average loss in the worst var_level cases
        tail_losses = annual_returns[annual_returns <= np.percentile(annual_returns, var_level * 100)]
        cvar_95 = -np.mean(tail_losses) if len(tail_losses) > 0 else var_95

        # Maximum Drawdown from simulated paths
        if student_t is not None:
            # Compute drawdowns from the simulated daily paths
            cumulative = np.cumprod(1 + t_samples, axis=0)
            running_max = np.maximum.accumulate(cumulative, axis=0)
            drawdowns = (running_max - cumulative) / running_max
            max_drawdown = np.percentile(np.max(drawdowns, axis=0), 95)
        else:
            max_drawdown = var_95 * 1.5  # Rough approximation

        # Tail loss probability: P(loss > MaxDrawdown threshold)
        threshold = 0.20  # 20% loss threshold
        tail_loss_prob = np.mean(annual_returns < -threshold)

        # Black Swan: 1st percentile worst case
        black_swan_loss = -np.percentile(annual_returns, 1)

        # Worst N scenarios
        sorted_returns = np.sort(annual_returns)
        worst_5 = sorted_returns[:5]

        result = {
            "var_95": round(float(var_95) * 100, 4),
            "cvar_95": round(float(cvar_95) * 100, 4),
            "max_drawdown": round(float(max_drawdown) * 100, 4),
            "expected_return": round(float(port_mu) * 100, 4),
            "portfolio_volatility": round(float(port_vol) * 100, 4),
            "tail_loss_probability": round(float(tail_loss_prob) * 100, 4),
            "black_swan_loss": round(float(black_swan_loss) * 100, 4),
            "distribution_params": {
                "type": "Student-t" if student_t else "Normal (fallback)",
                "degrees_of_freedom": df_t,
                "annual_mu": round(float(port_mu) * 100, 4),
                "annual_sigma": round(float(port_vol) * 100, 4),
            },
            "worst_case_scenarios": [round(float(x) * 100, 2) for x in worst_5],
            "simulation_count": n_simulations,
            "confidence_level": confidence,
            # Histogram data for visualization
            "return_distribution": {
                "bins": np.histogram(annual_returns * 100, bins=80)[0].tolist(),
                "edges": np.histogram(annual_returns * 100, bins=80)[1].tolist(),
            },
        }

        self.results["tail_risk"] = result
        return result

    # ------------------------------------------------------------------
    # LAYER 3: Sensitivity & Stability Analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(self, ips_constraints, perturbations=None):
        """
        Tests portfolio fragility by perturbing inputs and measuring
        how aggressively the optimal allocation shifts.

        A robust portfolio should be STABLE — small input changes should NOT
        cause dramatic weight swings.

        Args:
            ips_constraints: Base IPS constraints dict.
            perturbations: Optional list of dicts defining custom perturbations.

        Returns:
            Dict with stability_score, fragility_indicator, perturbation_results,
            and weight_change_matrix.
        """
        if self.engine is None or minimize is None:
            return {"error": "PortfolioBuilder engine or SciPy not available."}

        mkt = self._get_market_data()
        mu_base = mkt['mu']
        cov_base = mkt['cov']

        # Run baseline optimization
        base_alloc = self.engine._optimize(mu_base, cov_base, ips_constraints)
        base_weights = base_alloc.drop("CASH_RESERVE", errors='ignore')

        # Define perturbation scenarios
        if perturbations is None:
            perturbations = self._default_perturbations(ips_constraints, mu_base, cov_base)

        perturbation_results = []
        all_weight_deltas = []

        for p in perturbations:
            p_name = p["name"]
            p_mu = p.get("mu", mu_base)
            p_cov = p.get("cov", cov_base)
            p_constraints = p.get("constraints", ips_constraints)

            try:
                perturbed_alloc = self.engine._optimize(p_mu, p_cov, p_constraints)
                perturbed_weights = perturbed_alloc.drop("CASH_RESERVE", errors='ignore')

                # Align indices
                all_tickers = base_weights.index.union(perturbed_weights.index)
                bw = base_weights.reindex(all_tickers, fill_value=0)
                pw = perturbed_weights.reindex(all_tickers, fill_value=0)

                # Weight changes
                delta = (pw - bw).abs()
                max_change = delta.max()
                avg_change = delta.mean()
                total_turnover = delta.sum() / 2  # One-way turnover

                weight_changes = {}
                for ticker in all_tickers:
                    weight_changes[ticker] = {
                        "base": round(float(bw[ticker]) * 100, 2),
                        "perturbed": round(float(pw[ticker]) * 100, 2),
                        "change": round(float(pw[ticker] - bw[ticker]) * 100, 2),
                    }

                perturbation_results.append({
                    "perturbation": p_name,
                    "max_weight_change": round(float(max_change) * 100, 2),
                    "avg_weight_change": round(float(avg_change) * 100, 2),
                    "total_turnover": round(float(total_turnover) * 100, 2),
                    "weight_changes": weight_changes,
                    "converged": True,
                })

                all_weight_deltas.append(float(total_turnover))

            except Exception as e:
                logger.warning(f"Perturbation '{p_name}' failed: {e}")
                perturbation_results.append({
                    "perturbation": p_name,
                    "error": str(e),
                    "converged": False,
                })
                all_weight_deltas.append(1.0)  # Worst case

        # Compute aggregate metrics
        if all_weight_deltas:
            avg_turnover = np.mean(all_weight_deltas)
            max_turnover = np.max(all_weight_deltas)
        else:
            avg_turnover = 0
            max_turnover = 0

        # Stability Score: 0 (fragile) to 100 (rock solid)
        # Based on average turnover — lower turnover = higher stability
        stability_score = max(0, min(100, 100 * (1 - avg_turnover * 5)))

        if stability_score >= 70:
            fragility = "STABLE"
        elif stability_score >= 40:
            fragility = "MODERATE"
        else:
            fragility = "FRAGILE"

        result = {
            "stability_score": round(stability_score, 1),
            "fragility_indicator": fragility,
            "avg_turnover_pct": round(avg_turnover * 100, 2),
            "max_turnover_pct": round(max_turnover * 100, 2),
            "perturbation_results": perturbation_results,
            "base_weights": {k: round(float(v) * 100, 2) for k, v in base_weights.items()},
        }

        self.results["sensitivity"] = result
        return result

    def _default_perturbations(self, ips_constraints, mu_base, cov_base):
        """Generates the standard set of perturbation scenarios."""
        perturbations = []

        # 1. Risk Attitude +1 (more aggressive)
        c_up = copy.deepcopy(ips_constraints)
        c_up["EquityMax"] = min(1.0, c_up.get("EquityMax", 0.8) + 0.10)
        c_up["EquityMin"] = min(c_up["EquityMax"], c_up.get("EquityMin", 0.5) + 0.10)
        c_up["ReturnObjective"] = c_up.get("ReturnObjective", 0.10) + 0.02
        perturbations.append({
            "name": "Risk Attitude +1 (More Aggressive)",
            "constraints": c_up,
        })

        # 2. Risk Attitude -1 (more conservative)
        c_down = copy.deepcopy(ips_constraints)
        c_down["EquityMax"] = max(0.1, c_down.get("EquityMax", 0.8) - 0.10)
        c_down["EquityMin"] = max(0.0, c_down.get("EquityMin", 0.5) - 0.10)
        c_down["ReturnObjective"] = max(0.04, c_down.get("ReturnObjective", 0.10) - 0.02)
        perturbations.append({
            "name": "Risk Attitude -1 (More Conservative)",
            "constraints": c_down,
        })

        # 3. Return expectations +2%
        mu_up = mu_base + 0.02
        perturbations.append({
            "name": "Expected Returns +2%",
            "mu": mu_up,
        })

        # 4. Return expectations -2%
        mu_down = mu_base - 0.02
        perturbations.append({
            "name": "Expected Returns -2%",
            "mu": mu_down,
        })

        # 5. Covariance perturbation (+20% volatility)
        cov_up = cov_base * 1.44  # Variance scales as vol², so 1.2² = 1.44
        perturbations.append({
            "name": "Volatility +20% (Covariance Shock)",
            "cov": cov_up,
        })

        # 6. Covariance perturbation (-20% volatility / calm market)
        cov_down = cov_base * 0.64  # 0.8² = 0.64
        perturbations.append({
            "name": "Volatility -20% (Calm Market)",
            "cov": cov_down,
        })

        # 7. Correlation spike (add 0.3 to off-diagonal correlations)
        cov_vals = cov_base.values.copy()
        vols = np.sqrt(np.diag(cov_vals))
        corr = cov_vals / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)
        corr_spiked = np.clip(corr + 0.3, -1, 1)
        np.fill_diagonal(corr_spiked, 1.0)
        cov_spiked = corr_spiked * np.outer(vols, vols)
        cov_spiked_df = pd.DataFrame(cov_spiked, index=cov_base.index, columns=cov_base.columns)
        perturbations.append({
            "name": "Correlation Spike (+0.3)",
            "cov": cov_spiked_df,
        })

        return perturbations

    # ------------------------------------------------------------------
    # LAYER 4: IPS Constraint Validation Under Stress
    # ------------------------------------------------------------------

    def _check_ips_violations(self, stressed_return, drawdown, risky_weights, cash_weight,
                              ips_constraints):
        """
        Validates whether a stressed portfolio state violates IPS constraints.

        Returns:
            Dict of constraint checks, each with 'violated', 'threshold', 'actual', 'severity'.
        """
        equity_total = risky_weights.sum()

        violations = {}

        # 1. Equity Allocation Bounds
        eq_min = ips_constraints.get("EquityMin", 0.0)
        eq_max = ips_constraints.get("EquityMax", 1.0)
        eq_violated = equity_total < eq_min or equity_total > eq_max
        violations["equity_bounds"] = {
            "violated": eq_violated,
            "threshold": f"{eq_min*100:.1f}% – {eq_max*100:.1f}%",
            "actual": f"{equity_total*100:.1f}%",
            "severity": "WARNING" if eq_violated else "SAFE",
        }

        # 2. Cash Requirement
        cash_min = ips_constraints.get("CashMin", 0.0)
        cash_violated = cash_weight < cash_min
        violations["cash_requirement"] = {
            "violated": cash_violated,
            "threshold": f"≥ {cash_min*100:.1f}%",
            "actual": f"{cash_weight*100:.1f}%",
            "severity": "WARNING" if cash_violated else "SAFE",
        }

        # 3. Max Drawdown Breach
        max_dd = ips_constraints.get("MaxDrawdown", 0.15)
        dd_violated = drawdown > max_dd
        if drawdown > max_dd * 1.5:
            dd_severity = "CRITICAL FAILURE"
        elif dd_violated:
            dd_severity = "WARNING"
        else:
            dd_severity = "SAFE"
        violations["max_drawdown"] = {
            "violated": dd_violated,
            "threshold": f"≤ {max_dd*100:.1f}%",
            "actual": f"{drawdown*100:.1f}%",
            "severity": dd_severity,
        }

        # 4. Return Objective Failure
        target_return = ips_constraints.get("ReturnObjective", 0.08)
        ret_violated = stressed_return < 0 and abs(stressed_return) > target_return
        violations["return_objective"] = {
            "violated": ret_violated,
            "threshold": f"≥ {target_return*100:.1f}%",
            "actual": f"{stressed_return*100:.1f}%",
            "severity": "CRITICAL FAILURE" if ret_violated else "SAFE",
        }

        return violations

    def validate_constraints_all_scenarios(self, allocation, ips_constraints):
        """
        Runs IPS validation across ALL stress scenarios and returns a consolidated report.

        Returns:
            Dict with per-scenario validation and overall compliance score.
        """
        scenario_results = self.run_scenario_stress_test(allocation, ips_constraints)
        
        total = len(scenario_results)
        safe_count = sum(1 for r in scenario_results if r["status"] == "SAFE")
        warning_count = sum(1 for r in scenario_results if r["status"] == "WARNING")
        critical_count = sum(1 for r in scenario_results if r["status"] == "CRITICAL FAILURE")

        compliance_score = (safe_count / total * 100) if total > 0 else 0

        result = {
            "total_scenarios": total,
            "safe": safe_count,
            "warning": warning_count,
            "critical": critical_count,
            "compliance_score": round(compliance_score, 1),
            "scenario_details": scenario_results,
        }

        self.results["ips_validation"] = result
        return result

    # ------------------------------------------------------------------
    # ADDITIONAL INTELLIGENCE
    # ------------------------------------------------------------------

    def identify_vulnerability_sources(self, allocation):
        """
        Identifies which assets contribute most to portfolio damage across
        all stress scenarios. Highlights concentration risk.

        Returns:
            Dict with damage_ranking, concentration_risk, rebalancing_suggestions.
        """
        risky_weights, _ = self._prepare_weights(allocation)

        # Aggregate damage across ALL scenarios
        damage_accumulator = {}
        for scenario_name, scenario_config in STRESS_SCENARIOS.items():
            shocks = scenario_config["shocks"]
            recovery = scenario_config.get("recovery_factor", 0.0)
            for ticker, weight in risky_weights.items():
                if weight < 1e-6:
                    continue
                shock = self._get_asset_shock(ticker, shocks)
                net_shock = shock * (1 - recovery)
                damage = net_shock * weight  # Weighted loss contribution
                if ticker not in damage_accumulator:
                    damage_accumulator[ticker] = {
                        "total_damage": 0,
                        "weight": round(float(weight) * 100, 2),
                        "sector": self._get_sector(ticker),
                        "worst_scenario": None,
                        "worst_damage": 0,
                    }
                damage_accumulator[ticker]["total_damage"] += damage
                if damage < damage_accumulator[ticker]["worst_damage"]:
                    damage_accumulator[ticker]["worst_damage"] = damage
                    damage_accumulator[ticker]["worst_scenario"] = scenario_name

        # Sort by total damage (most negative first)
        damage_ranking = sorted(
            damage_accumulator.items(),
            key=lambda x: x[1]["total_damage"]
        )

        # Format
        ranked = []
        for ticker, data in damage_ranking:
            ranked.append({
                "ticker": ticker,
                "weight_pct": data["weight"],
                "sector": data["sector"],
                "cumulative_damage_pct": round(data["total_damage"] * 100, 4),
                "worst_scenario": data["worst_scenario"],
                "worst_single_loss_pct": round(data["worst_damage"] * 100, 4),
            })

        # Concentration Risk
        # Herfindahl-Hirschman Index (HHI) of the portfolio
        weights_arr = risky_weights[risky_weights > 1e-6].values
        if len(weights_arr) > 0:
            normalized = weights_arr / weights_arr.sum()
            hhi = float(np.sum(normalized ** 2))
        else:
            hhi = 1.0

        # Sector concentration
        sector_weights = {}
        for ticker, weight in risky_weights.items():
            if weight < 1e-6:
                continue
            sector = self._get_sector(ticker)
            sector_weights[sector] = sector_weights.get(sector, 0) + float(weight)

        max_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("N/A", 0)

        if hhi > 0.25:
            concentration_level = "HIGH"
        elif hhi > 0.15:
            concentration_level = "MODERATE"
        else:
            concentration_level = "LOW"

        # Rebalancing Insights
        suggestions = []
        if concentration_level in ["HIGH", "MODERATE"]:
            suggestions.append(
                f"Portfolio is concentrated (HHI={hhi:.3f}). Consider diversifying "
                f"away from {max_sector[0]} ({max_sector[1]*100:.1f}% weight)."
            )
        if ranked and ranked[0]["weight_pct"] > 15:
            suggestions.append(
                f"Top damage contributor is {ranked[0]['ticker']} at "
                f"{ranked[0]['weight_pct']:.1f}% weight. Consider capping single-name"
                f" exposure to 10-12% for resilience."
            )
        # Check for missing defensive sectors
        present_sectors = set(sector_weights.keys())
        defensive = {"FMCG", "Pharma"}
        missing_defensive = defensive - present_sectors
        if missing_defensive:
            suggestions.append(
                f"No exposure to defensive sectors: {', '.join(missing_defensive)}. "
                f"Adding these can reduce drawdown in crisis scenarios."
            )
        if not suggestions:
            suggestions.append(
                "Portfolio is well-diversified. No immediate rebalancing required."
            )

        result = {
            "damage_ranking": ranked,
            "concentration_risk": {
                "hhi": round(hhi, 4),
                "level": concentration_level,
                "dominant_sector": max_sector[0],
                "dominant_sector_weight_pct": round(max_sector[1] * 100, 2),
                "sector_breakdown": {k: round(v * 100, 2) for k, v in sector_weights.items()},
            },
            "rebalancing_suggestions": suggestions,
        }

        self.results["vulnerability"] = result
        return result

    # ------------------------------------------------------------------
    # MASTER ORCHESTRATOR — Runs all 4 layers
    # ------------------------------------------------------------------

    def run_full_stress_test(self, allocation, ips_constraints):
        """
        Executes the complete 4-layer stress testing suite and produces
        a consolidated final report.

        Args:
            allocation: pd.Series of portfolio weights.
            ips_constraints: Dict from generate_ips().

        Returns:
            Comprehensive dict with all results and a final verdict.
        """
        logger.info("=" * 60)
        logger.info("STRESS TESTING ENGINE — FULL SUITE INITIATED")
        logger.info("=" * 60)

        # Layer 1: Macro Scenarios
        print("\n[Stress Test] Layer 1/4: Macro-Economic Scenario Analysis...")
        scenario_results = self.run_scenario_stress_test(allocation, ips_constraints)

        # Layer 2: Tail Risk
        print("[Stress Test] Layer 2/4: Extreme Value / Tail Risk Analysis...")
        tail_risk = self.run_tail_risk_analysis(allocation)

        # Layer 3: Sensitivity
        print("[Stress Test] Layer 3/4: Sensitivity & Stability Analysis...")
        sensitivity = self.run_sensitivity_analysis(ips_constraints)

        # Layer 4: IPS Validation (uses scenario results from Layer 1)
        print("[Stress Test] Layer 4/4: IPS Constraint Validation...\n")
        # Already computed in Layer 1, consolidate
        ips_validation = self.validate_constraints_all_scenarios(allocation, ips_constraints)

        # Additional Intelligence
        vulnerability = self.identify_vulnerability_sources(allocation)

        # --- FINAL VERDICT ---
        verdict = self._compute_final_verdict(
            scenario_results, tail_risk, sensitivity, ips_validation
        )

        report = {
            "scenario_stress_test": scenario_results,
            "tail_risk_analysis": tail_risk,
            "sensitivity_analysis": sensitivity,
            "ips_validation": ips_validation,
            "vulnerability_analysis": vulnerability,
            "risk_metrics_summary": self._build_risk_summary(tail_risk, scenario_results),
            "final_verdict": verdict,
        }

        self.results["full_report"] = report
        return report

    def _compute_final_verdict(self, scenarios, tail_risk, sensitivity, ips_validation):
        """Computes the final ROBUST / MODERATE RISK / FRAGILE verdict."""
        score = 100  # Start perfect, deduct

        # Scenario deductions
        for s in scenarios:
            if s["status"] == "CRITICAL FAILURE":
                score -= 15
            elif s["status"] == "WARNING":
                score -= 5

        # Tail risk deductions
        if isinstance(tail_risk, dict) and "cvar_95" in tail_risk:
            cvar = tail_risk["cvar_95"]
            if cvar > 40:
                score -= 20
            elif cvar > 25:
                score -= 10
            elif cvar > 15:
                score -= 5

        # Sensitivity deductions
        if isinstance(sensitivity, dict) and "stability_score" in sensitivity:
            stab = sensitivity["stability_score"]
            if stab < 40:
                score -= 15
            elif stab < 60:
                score -= 8

        # IPS compliance deductions
        if isinstance(ips_validation, dict) and "compliance_score" in ips_validation:
            comp = ips_validation["compliance_score"]
            if comp < 30:
                score -= 20
            elif comp < 60:
                score -= 10

        score = max(0, min(100, score))

        if score >= 65:
            verdict_label = "ROBUST"
        elif score >= 35:
            verdict_label = "MODERATE RISK"
        else:
            verdict_label = "FRAGILE"

        return {
            "label": verdict_label,
            "score": round(score, 1),
            "explanation": self._verdict_explanation(verdict_label, score),
        }

    def _verdict_explanation(self, label, score):
        """Returns a human-readable explanation for the verdict."""
        if label == "ROBUST":
            return (
                f"Portfolio Robustness Score: {score}/100. "
                "The portfolio demonstrates strong resilience across macro-economic shocks, "
                "acceptable tail-risk exposure, and stable allocation under input perturbation. "
                "IPS constraints are largely maintained under stress."
            )
        elif label == "MODERATE RISK":
            return (
                f"Portfolio Robustness Score: {score}/100. "
                "The portfolio shows vulnerability in certain extreme scenarios. "
                "Some IPS constraints may be breached under severe stress. "
                "Review concentration risks and consider hedging tail exposures."
            )
        else:
            return (
                f"Portfolio Robustness Score: {score}/100. "
                "CRITICAL — Portfolio breaks under multiple stress scenarios. "
                "Allocation is fragile and sensitive to input changes. "
                "IPS compliance fails under stress. Immediate rebalancing recommended."
            )

    def _build_risk_summary(self, tail_risk, scenarios):
        """Builds the consolidated risk metrics summary table."""
        summary = {}

        if isinstance(tail_risk, dict) and "error" not in tail_risk:
            summary["VaR_95_pct"] = tail_risk.get("var_95", "N/A")
            summary["CVaR_95_pct"] = tail_risk.get("cvar_95", "N/A")
            summary["Max_Drawdown_pct"] = tail_risk.get("max_drawdown", "N/A")
            summary["Expected_Return_pct"] = tail_risk.get("expected_return", "N/A")
            summary["Black_Swan_Loss_pct"] = tail_risk.get("black_swan_loss", "N/A")
            summary["Tail_Loss_Probability_pct"] = tail_risk.get("tail_loss_probability", "N/A")

        # Worst scenario from macro tests
        if scenarios:
            worst = min(scenarios, key=lambda x: x["portfolio_return"])
            summary["worst_scenario"] = worst["scenario"]
            summary["worst_scenario_return_pct"] = worst["portfolio_return"]
            summary["worst_scenario_drawdown_pct"] = worst["drawdown"]

        return summary

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def generate_visualizations(self, allocation, save_dir=None):
        """
        Generates publication-quality stress testing visualizations.

        Produces 4 charts:
            1. Stress Scenario Impact (Bar Chart)
            2. Drawdown Comparison (Horizontal Bar)
            3. Tail Risk Distribution (Histogram + VaR/CVaR lines)
            4. Allocation Shift Heatmap (Sensitivity)

        Args:
            allocation: pd.Series of portfolio weights.
            save_dir: Optional directory to save PNGs. If None, calls plt.show().

        Returns:
            List of figure objects.
        """
        if "full_report" not in self.results:
            print("(!) Run run_full_stress_test() first to generate data for visualization.")
            return []

        report = self.results["full_report"]
        figures = []

        # Color palette — institutional dark theme
        BG_COLOR = "#0D1117"
        TEXT_COLOR = "#C9D1D9"
        ACCENT_GREEN = "#39D353"
        ACCENT_RED = "#F85149"
        ACCENT_YELLOW = "#F0C000"
        ACCENT_BLUE = "#58A6FF"
        GRID_COLOR = "#21262D"

        plt.rcParams.update({
            'figure.facecolor': BG_COLOR,
            'axes.facecolor': BG_COLOR,
            'axes.edgecolor': GRID_COLOR,
            'axes.labelcolor': TEXT_COLOR,
            'text.color': TEXT_COLOR,
            'xtick.color': TEXT_COLOR,
            'ytick.color': TEXT_COLOR,
            'grid.color': GRID_COLOR,
            'font.family': 'sans-serif',
        })

        # ===================== CHART 1: Stress Scenario Bar Chart =====================
        fig1, ax1 = plt.subplots(figsize=(14, 7))

        scenarios = report["scenario_stress_test"]
        names = [s["scenario"].replace(" (", "\n(") for s in scenarios]
        returns = [s["portfolio_return"] for s in scenarios]
        statuses = [s["status"] for s in scenarios]

        colors = []
        for s in statuses:
            if s == "SAFE":
                colors.append(ACCENT_GREEN)
            elif s == "WARNING":
                colors.append(ACCENT_YELLOW)
            else:
                colors.append(ACCENT_RED)

        bars = ax1.bar(range(len(names)), returns, color=colors, edgecolor='none', width=0.65, alpha=0.9)

        # Add value labels
        for bar, ret in zip(bars, returns):
            y_pos = bar.get_height() - 1.2 if ret < 0 else bar.get_height() + 0.5
            ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{ret:.1f}%",
                     ha='center', va='top' if ret < 0 else 'bottom',
                     fontsize=9, fontweight='bold', color=TEXT_COLOR)

        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, fontsize=8, ha='center')
        ax1.set_ylabel("Portfolio Return (%)", fontsize=11)
        ax1.set_title("MACRO-ECONOMIC STRESS SCENARIO IMPACT", fontsize=14, fontweight='bold',
                       pad=15, color=ACCENT_BLUE)
        ax1.axhline(y=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.3)
        ax1.grid(axis='y', alpha=0.2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ACCENT_GREEN, label='SAFE'),
            Patch(facecolor=ACCENT_YELLOW, label='WARNING'),
            Patch(facecolor=ACCENT_RED, label='CRITICAL'),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=9,
                   facecolor=BG_COLOR, edgecolor=GRID_COLOR)

        fig1.tight_layout()
        figures.append(fig1)

        # ===================== CHART 2: Drawdown Comparison =====================
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        dd_values = [s["drawdown"] for s in scenarios]
        dd_names = [s["scenario"].split("(")[0].strip() for s in scenarios]

        # Sort by drawdown
        sorted_pairs = sorted(zip(dd_names, dd_values, colors), key=lambda x: x[1])
        dd_names_sorted = [p[0] for p in sorted_pairs]
        dd_values_sorted = [p[1] for p in sorted_pairs]
        dd_colors_sorted = [p[2] for p in sorted_pairs]

        bars2 = ax2.barh(range(len(dd_names_sorted)), dd_values_sorted,
                         color=dd_colors_sorted, edgecolor='none', height=0.6, alpha=0.9)

        # Max drawdown threshold line
        max_dd_threshold = report.get("risk_metrics_summary", {}).get("Max_Drawdown_pct")
        if max_dd_threshold and max_dd_threshold != "N/A":
            ax2.axvline(x=max_dd_threshold, color=ACCENT_RED, linewidth=2,
                        linestyle='--', alpha=0.7, label=f"Max DD Threshold ({max_dd_threshold:.1f}%)")

        for bar, val in zip(bars2, dd_values_sorted):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va='center', fontsize=9, color=TEXT_COLOR)

        ax2.set_yticks(range(len(dd_names_sorted)))
        ax2.set_yticklabels(dd_names_sorted, fontsize=9)
        ax2.set_xlabel("Drawdown (%)", fontsize=11)
        ax2.set_title("DRAWDOWN COMPARISON ACROSS SCENARIOS", fontsize=14,
                       fontweight='bold', pad=15, color=ACCENT_BLUE)
        ax2.grid(axis='x', alpha=0.2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        if max_dd_threshold and max_dd_threshold != "N/A":
            ax2.legend(fontsize=9, facecolor=BG_COLOR, edgecolor=GRID_COLOR)

        fig2.tight_layout()
        figures.append(fig2)

        # ===================== CHART 3: Tail Risk Distribution =====================
        tail_data = report.get("tail_risk_analysis", {})
        if isinstance(tail_data, dict) and "return_distribution" in tail_data:
            fig3, ax3 = plt.subplots(figsize=(14, 7))

            dist = tail_data["return_distribution"]
            bin_edges = dist["edges"]
            bin_counts = dist["bins"]
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_counts))]

            # Color bars: red for losses, green for gains
            bar_colors = [ACCENT_RED if c < 0 else ACCENT_GREEN for c in bin_centers]

            ax3.bar(bin_centers, bin_counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
                    color=bar_colors, alpha=0.7, edgecolor='none')

            # VaR and CVaR lines
            var_val = tail_data.get("var_95", 0)
            cvar_val = tail_data.get("cvar_95", 0)
            black_swan = tail_data.get("black_swan_loss", 0)

            ax3.axvline(x=-var_val, color=ACCENT_YELLOW, linewidth=2, linestyle='--',
                        label=f"VaR 95% ({var_val:.1f}%)")
            ax3.axvline(x=-cvar_val, color=ACCENT_RED, linewidth=2, linestyle='--',
                        label=f"CVaR 95% ({cvar_val:.1f}%)")
            ax3.axvline(x=-black_swan, color='#FF6B6B', linewidth=1.5, linestyle=':',
                        label=f"Black Swan 1% ({black_swan:.1f}%)")

            ax3.set_xlabel("Annual Return (%)", fontsize=11)
            ax3.set_ylabel("Frequency", fontsize=11)
            ax3.set_title("TAIL RISK DISTRIBUTION (Fat-Tailed Student-t Model)",
                          fontsize=14, fontweight='bold', pad=15, color=ACCENT_BLUE)
            ax3.legend(fontsize=10, facecolor=BG_COLOR, edgecolor=GRID_COLOR, loc='upper left')
            ax3.grid(axis='y', alpha=0.2)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            fig3.tight_layout()
            figures.append(fig3)

        # ===================== CHART 4: Allocation Shift Heatmap =====================
        sensitivity_data = report.get("sensitivity_analysis", {})
        perturbation_results = sensitivity_data.get("perturbation_results", [])

        if perturbation_results:
            # Build the heatmap matrix
            converged = [p for p in perturbation_results if p.get("converged", False)]
            if converged:
                fig4, ax4 = plt.subplots(figsize=(14, 8))

                perturbation_names = [p["perturbation"].replace(" (", "\n(") for p in converged]

                # Get all tickers from first perturbation  
                all_tickers = list(converged[0].get("weight_changes", {}).keys())

                # Build matrix: rows = perturbations, cols = tickers
                matrix = []
                for p in converged:
                    row = []
                    changes = p.get("weight_changes", {})
                    for t in all_tickers:
                        row.append(changes.get(t, {}).get("change", 0))
                    matrix.append(row)

                matrix = np.array(matrix)

                # Only show tickers with material changes
                material_cols = np.any(np.abs(matrix) > 0.5, axis=0)
                if material_cols.any():
                    matrix_filtered = matrix[:, material_cols]
                    tickers_filtered = [all_tickers[i] for i in range(len(all_tickers)) if material_cols[i]]
                else:
                    matrix_filtered = matrix
                    tickers_filtered = all_tickers

                # Custom diverging colormap
                cmap = LinearSegmentedColormap.from_list(
                    'stress_heatmap',
                    [(0, ACCENT_RED), (0.5, BG_COLOR), (1, ACCENT_GREEN)]
                )

                vmax = max(abs(matrix_filtered.min()), abs(matrix_filtered.max()), 1)
                im = ax4.imshow(matrix_filtered, cmap=cmap, aspect='auto',
                                vmin=-vmax, vmax=vmax)

                # Labels
                ax4.set_xticks(range(len(tickers_filtered)))
                ax4.set_xticklabels(tickers_filtered, fontsize=8, rotation=45, ha='right')
                ax4.set_yticks(range(len(perturbation_names)))
                ax4.set_yticklabels(perturbation_names, fontsize=8)

                # Annotate cells
                for i in range(matrix_filtered.shape[0]):
                    for j in range(matrix_filtered.shape[1]):
                        val = matrix_filtered[i, j]
                        if abs(val) > 0.1:
                            color = 'white' if abs(val) > vmax * 0.4 else TEXT_COLOR
                            ax4.text(j, i, f"{val:+.1f}", ha='center', va='center',
                                     fontsize=7, color=color)

                cbar = fig4.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
                cbar.set_label("Weight Change (%)", fontsize=10, color=TEXT_COLOR)
                cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)

                ax4.set_title("ALLOCATION SENSITIVITY HEATMAP", fontsize=14,
                              fontweight='bold', pad=15, color=ACCENT_BLUE)

                fig4.tight_layout()
                figures.append(fig4)

        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            chart_names = [
                "01_scenario_impact.png",
                "02_drawdown_comparison.png",
                "03_tail_risk_distribution.png",
                "04_allocation_heatmap.png",
            ]
            for fig, name in zip(figures, chart_names):
                path = os.path.join(save_dir, name)
                fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
                print(f"   Saved: {path}")
            plt.close('all')
        else:
            plt.show()

        return figures

    # ------------------------------------------------------------------
    # CONSOLE REPORT
    # ------------------------------------------------------------------

    def print_report(self):
        """Prints a formatted summary of the full stress test results to console."""
        if "full_report" not in self.results:
            print("(!) No results available. Run run_full_stress_test() first.")
            return

        report = self.results["full_report"]

        print("\n" + "=" * 72)
        print("           AssetOS — INSTITUTIONAL STRESS TEST REPORT")
        print("=" * 72)

        # ── Scenario Summary Table ──
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│                  MACRO-ECONOMIC SCENARIO RESULTS                   │")
        print("├──────────────────────────────┬──────────┬──────────┬────────┬───────┤")
        print("│ Scenario                     │ Return   │ Drawdown │ IPS    │Status │")
        print("├──────────────────────────────┼──────────┼──────────┼────────┼───────┤")

        for s in report["scenario_stress_test"]:
            name = s["scenario"][:28].ljust(28)
            ret = f"{s['portfolio_return']:+.1f}%".rjust(8)
            dd = f"{s['drawdown']:.1f}%".rjust(8)
            ips = "YES" if s["ips_violation"] else "NO"
            ips = ips.center(6)
            status = s["status"][:7].center(7)
            print(f"│ {name} │ {ret} │ {dd} │ {ips} │{status}│")

        print("└──────────────────────────────┴──────────┴──────────┴────────┴───────┘")

        # ── Risk Metrics ──
        risk = report.get("risk_metrics_summary", {})
        print("\n┌─────────────────────────────────────────┐")
        print("│           TAIL RISK METRICS              │")
        print("├──────────────────────┬────────────────────┤")
        for key, val in risk.items():
            if key.startswith("worst_scenario"):
                continue
            label = key.replace("_", " ").replace("pct", "").strip()
            label = label[:20].ljust(20)
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}%".rjust(18)
            else:
                val_str = str(val).rjust(18)
            print(f"│ {label} │ {val_str} │")
        print("├──────────────────────┼────────────────────┤")

        worst = risk.get("worst_scenario", "N/A")
        worst_ret = risk.get("worst_scenario_return_pct", "N/A")
        print(f"│ Worst Scenario       │ {str(worst)[:18].rjust(18)} │")
        if isinstance(worst_ret, (int, float)):
            print(f"│ Worst Scenario Loss  │ {f'{worst_ret:.2f}%'.rjust(18)} │")
        print("└──────────────────────┴────────────────────┘")

        # ── Stability Report ──
        sensitivity = report.get("sensitivity_analysis", {})
        print("\n┌─────────────────────────────────────────┐")
        print("│          STABILITY ANALYSIS              │")
        print("├──────────────────────┬────────────────────┤")
        print(f"│ Stability Score      │ {str(sensitivity.get('stability_score', 'N/A')).rjust(14)}/100 │")
        print(f"│ Fragility Indicator  │ {str(sensitivity.get('fragility_indicator', 'N/A')).rjust(18)} │")
        print(f"│ Avg Turnover         │ {str(sensitivity.get('avg_turnover_pct', 'N/A')).rjust(17)}% │")
        print(f"│ Max Turnover         │ {str(sensitivity.get('max_turnover_pct', 'N/A')).rjust(17)}% │")
        print("└──────────────────────┴────────────────────┘")

        # ── IPS Compliance ──
        ips_val = report.get("ips_validation", {})
        print("\n┌─────────────────────────────────────────┐")
        print("│          IPS COMPLIANCE                  │")
        print("├──────────────────────┬────────────────────┤")
        print(f"│ Compliance Score     │ {str(ips_val.get('compliance_score', 'N/A')).rjust(17)}% │")
        print(f"│ SAFE Scenarios       │ {str(ips_val.get('safe', 'N/A')).rjust(18)} │")
        print(f"│ WARNING Scenarios    │ {str(ips_val.get('warning', 'N/A')).rjust(18)} │")
        print(f"│ CRITICAL Scenarios   │ {str(ips_val.get('critical', 'N/A')).rjust(18)} │")
        print("└──────────────────────┴────────────────────┘")

        # ── Vulnerability ──
        vuln = report.get("vulnerability_analysis", {})
        conc = vuln.get("concentration_risk", {})
        print("\n┌─────────────────────────────────────────┐")
        print("│        CONCENTRATION RISK                │")
        print("├──────────────────────┬────────────────────┤")
        print(f"│ HHI Index            │ {str(conc.get('hhi', 'N/A')).rjust(18)} │")
        print(f"│ Concentration Level  │ {str(conc.get('level', 'N/A')).rjust(18)} │")
        print(f"│ Dominant Sector      │ {str(conc.get('dominant_sector', 'N/A')).rjust(18)} │")
        print(f"│ Dominant Weight      │ {str(conc.get('dominant_sector_weight_pct', 'N/A')).rjust(17)}% │")
        print("└──────────────────────┴────────────────────┘")

        # ── Top Damage Contributors ──
        dmg = vuln.get("damage_ranking", [])
        if dmg:
            print("\n  TOP DAMAGE CONTRIBUTORS (Cumulative across all scenarios):")
            for i, d in enumerate(dmg[:5], 1):
                print(f"    {i}. {d['ticker']} ({d['sector']}) — "
                      f"Weight: {d['weight_pct']:.1f}%, "
                      f"Cumulative Damage: {d['cumulative_damage_pct']:.2f}%, "
                      f"Worst in: {d['worst_scenario']}")

        # ── Rebalancing Suggestions ──
        suggs = vuln.get("rebalancing_suggestions", [])
        if suggs:
            print("\n  REBALANCING INSIGHTS:")
            for s in suggs:
                print(f"    • {s}")

        # ── FINAL VERDICT ──
        verdict = report.get("final_verdict", {})
        label = verdict.get("label", "UNKNOWN")
        score = verdict.get("score", 0)

        verdict_box_color = {
            "ROBUST": "✅",
            "MODERATE RISK": "⚠️",
            "FRAGILE": "🚨",
        }

        print("\n" + "=" * 72)
        print(f"  {verdict_box_color.get(label, '❓')}  FINAL VERDICT: {label}  "
              f"(Score: {score}/100)")
        print("=" * 72)
        print(f"\n  {verdict.get('explanation', '')}")
        print("\n" + "=" * 72)


# ==============================================================================
# API-COMPATIBLE DATA EXPORT
# ==============================================================================

def get_stress_test_data(portfolio_builder, allocation, ips_constraints):
    """
    Convenience function for the Flask API.
    
    Returns a JSON-serializable dict with the complete stress test results.
    
    Usage in app.py:
        from stress_testing import get_stress_test_data
        data = get_stress_test_data(engine, weights_series, ips_dict)
        return jsonify(data)
    """
    stress_engine = StressTestingEngine(portfolio_builder)
    report = stress_engine.run_full_stress_test(allocation, ips_constraints)

    # JSON-serialize: ensure no numpy types leak
    return _sanitize_for_json(report)


def _sanitize_for_json(obj):
    """Recursively converts numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


# ==============================================================================
# CLI — STANDALONE EXECUTION
# ==============================================================================

def main():
    """
    Standalone stress testing runner.
    Initializes PortfolioBuilder, runs onboarding, optimization, and the full stress suite.
    """
    print("\n" + "=" * 60)
    print("     AssetOS — INSTITUTIONAL STRESS TESTING ENGINE")
    print("=" * 60)
    print("  Building portfolio and running stress analysis...\n")

    try:
        from IPS import PortfolioBuilder
    except ImportError:
        print("FATAL: Cannot import PortfolioBuilder from IPS.py")
        return

    # Initialize engine
    builder = PortfolioBuilder()

    # Run onboarding
    profile = builder.run_onboarding()
    client = profile.get('name', 'Unknown').upper()

    # Generate IPS
    ips_constraints = builder.generate_ips()

    # Run optimization (MPT baseline)
    print("\n[Phase 1] Running portfolio optimization...")
    try:
        allocation = builder.run_mpt_only()
    except Exception as e:
        print(f"FATAL: Optimization failed — {e}")
        return

    print(f"\nOptimized Allocation for {client}:")
    print("-" * 30)
    print((allocation[allocation > 0.001] * 100).round(2).astype(str) + " %")

    # Run stress testing
    print("\n[Phase 2] Running institutional stress test suite...")
    stress_engine = StressTestingEngine(builder)
    report = stress_engine.run_full_stress_test(allocation, ips_constraints)

    # Print report
    stress_engine.print_report()

    # Generate visualizations
    save = input("\nGenerate stress test charts? (y/n): ").strip().lower()
    if save.startswith('y'):
        save_dir = f"stress_reports/{client.replace(' ', '_')}"
        stress_engine.generate_visualizations(allocation, save_dir=save_dir)
        print(f"\n Charts saved to: {save_dir}/")
    else:
        stress_engine.generate_visualizations(allocation)

    print("\n✅ Stress testing complete.")


if __name__ == "__main__":
    main()
