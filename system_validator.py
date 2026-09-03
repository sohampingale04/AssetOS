"""
================================================================================
AssetOS — INSTITUTIONAL-GRADE SYSTEM VALIDATION & STRESS TESTING ENGINE
================================================================================

Black-box interrogation system that generates extreme investor profiles, runs
each through the full AssetOS pipeline (IPS → Optimization → Stress Testing),
and then aggressively attacks the outputs with 8 failure-detection tests.

Produces:
    • Per-investor detailed audit results
    • System-level diagnostic metrics
    • Final ROBUST / MODERATE / FRAGILE verdict
    • 5 publication-quality visualizations
    • Full CSV export

Usage:
    python system_validator.py

Author: AssetOS Validation Engine
================================================================================
"""

import os
import sys
import copy
import time
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from itertools import combinations

# Suppress noisy warnings during batch runs
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import AssetOS core modules
try:
    from IPS import PortfolioBuilder, CONFIG
    from stress_testing import StressTestingEngine, SECTOR_MAP
except ImportError as e:
    print(f"FATAL: Cannot import AssetOS modules — {e}")
    print("Ensure IPS.py and stress_testing.py are in the same directory.")
    sys.exit(1)

# Override data directory to use AssetOS_Data
CONFIG['data_dir'] = "AssetOS_Data"


# ==============================================================================
# CONFIGURATION
# ==============================================================================

OUTPUT_DIR = "validation_results"
CSV_FILENAME = "system_audit.csv"
RISK_FREE_RATE = CONFIG.get('risk_free_rate', 0.065)

# Visualization theme — institutional dark
THEME = {
    'bg': '#0D1117',
    'text': '#C9D1D9',
    'green': '#39D353',
    'red': '#F85149',
    'yellow': '#F0C000',
    'blue': '#58A6FF',
    'purple': '#BC8CFF',
    'orange': '#F78166',
    'grid': '#21262D',
    'accent': '#1F6FEB',
}


# ==============================================================================
# MODULE 1: EXTREME & DIVERSE INVESTOR PROFILES
# ==============================================================================

EXTREME_PROFILES = [
    # ── Normal Profiles ──
    {
        "name": "Conservative Retiree",
        "age": 62, "income": 600000, "net_worth": 15000000,
        "capital": 5000000, "goal": "Preservation", "goal_years": 5,
        "risk_attitude": 2, "loss_tolerance": 8, "liquidity": 15,
    },
    {
        "name": "Moderate Professional",
        "age": 38, "income": 2500000, "net_worth": 8000000,
        "capital": 3000000, "goal": "Wealth", "goal_years": 15,
        "risk_attitude": 3, "loss_tolerance": 15, "liquidity": 10,
    },
    {
        "name": "Aggressive Young Investor",
        "age": 26, "income": 1800000, "net_worth": 3000000,
        "capital": 1500000, "goal": "Wealth", "goal_years": 20,
        "risk_attitude": 5, "loss_tolerance": 30, "liquidity": 5,
    },

    # ── Ultra Risk-Averse ──
    {
        "name": "Ultra Conservative A",
        "age": 70, "income": 300000, "net_worth": 20000000,
        "capital": 8000000, "goal": "Preservation", "goal_years": 3,
        "risk_attitude": 1, "loss_tolerance": 2, "liquidity": 20,
    },
    {
        "name": "Ultra Conservative B",
        "age": 65, "income": 500000, "net_worth": 12000000,
        "capital": 4000000, "goal": "Income", "goal_years": 5,
        "risk_attitude": 1, "loss_tolerance": 5, "liquidity": 15,
    },

    # ── Ultra Aggressive ──
    {
        "name": "Max Risk Seeker A",
        "age": 22, "income": 3000000, "net_worth": 5000000,
        "capital": 2500000, "goal": "Wealth", "goal_years": 25,
        "risk_attitude": 5, "loss_tolerance": 40, "liquidity": 2,
    },
    {
        "name": "Max Risk Seeker B",
        "age": 28, "income": 4000000, "net_worth": 8000000,
        "capital": 4000000, "goal": "Wealth", "goal_years": 20,
        "risk_attitude": 5, "loss_tolerance": 35, "liquidity": 3,
    },

    # ── High Liquidity ──
    {
        "name": "High Liquidity Investor A",
        "age": 45, "income": 2000000, "net_worth": 10000000,
        "capital": 3000000, "goal": "Preservation", "goal_years": 10,
        "risk_attitude": 3, "loss_tolerance": 12, "liquidity": 40,
    },
    {
        "name": "High Liquidity Investor B",
        "age": 50, "income": 1500000, "net_worth": 7000000,
        "capital": 2000000, "goal": "Income", "goal_years": 8,
        "risk_attitude": 2, "loss_tolerance": 10, "liquidity": 60,
    },

    # ── Low Liquidity ──
    {
        "name": "Zero Liquidity Investor",
        "age": 30, "income": 3000000, "net_worth": 6000000,
        "capital": 2500000, "goal": "Wealth", "goal_years": 20,
        "risk_attitude": 4, "loss_tolerance": 25, "liquidity": 0,
    },
    {
        "name": "Minimal Liquidity Investor",
        "age": 35, "income": 2200000, "net_worth": 5000000,
        "capital": 2000000, "goal": "Wealth", "goal_years": 15,
        "risk_attitude": 4, "loss_tolerance": 20, "liquidity": 2,
    },

    # ── Short Horizon ──
    {
        "name": "Ultra Short Horizon (1yr)",
        "age": 55, "income": 1000000, "net_worth": 9000000,
        "capital": 3000000, "goal": "Preservation", "goal_years": 1,
        "risk_attitude": 2, "loss_tolerance": 5, "liquidity": 20,
    },
    {
        "name": "Short Horizon (2yr)",
        "age": 48, "income": 1500000, "net_worth": 7000000,
        "capital": 2500000, "goal": "Income", "goal_years": 2,
        "risk_attitude": 3, "loss_tolerance": 10, "liquidity": 15,
    },

    # ── Very Long Horizon ──
    {
        "name": "Long Horizon 25yr",
        "age": 25, "income": 1200000, "net_worth": 2000000,
        "capital": 1000000, "goal": "Wealth", "goal_years": 25,
        "risk_attitude": 4, "loss_tolerance": 25, "liquidity": 5,
    },
    {
        "name": "Ultra Long Horizon 30yr",
        "age": 22, "income": 800000, "net_worth": 1500000,
        "capital": 500000, "goal": "Wealth", "goal_years": 30,
        "risk_attitude": 5, "loss_tolerance": 30, "liquidity": 3,
    },

    # ── Mismatch Profiles ──
    {
        "name": "Low Capital High NW",
        "age": 55, "income": 5000000, "net_worth": 50000000,
        "capital": 200000, "goal": "Preservation", "goal_years": 10,
        "risk_attitude": 2, "loss_tolerance": 8, "liquidity": 10,
    },
    {
        "name": "High Capital Low Income",
        "age": 40, "income": 300000, "net_worth": 2000000,
        "capital": 10000000, "goal": "Wealth", "goal_years": 15,
        "risk_attitude": 3, "loss_tolerance": 15, "liquidity": 10,
    },
    {
        "name": "Young Retiree Paradox",
        "age": 28, "income": 200000, "net_worth": 1000000,
        "capital": 500000, "goal": "Preservation", "goal_years": 3,
        "risk_attitude": 1, "loss_tolerance": 3, "liquidity": 25,
    },

    # ── Unrealistic / Adversarial ──
    {
        "name": "Contradiction Profile",
        "age": 25, "income": 10000000, "net_worth": 100000000,
        "capital": 50000000, "goal": "Preservation", "goal_years": 1,
        "risk_attitude": 5, "loss_tolerance": 40, "liquidity": 50,
    },
    {
        "name": "Extreme Edge Case",
        "age": 80, "income": 100000, "net_worth": 500000,
        "capital": 100000, "goal": "Income", "goal_years": 30,
        "risk_attitude": 5, "loss_tolerance": 40, "liquidity": 0,
    },
]


# ==============================================================================
# MODULE 2: FULL PIPELINE RUNNER
# ==============================================================================

def run_full_pipeline(profiles):
    """
    Runs each investor profile through the complete AssetOS pipeline:
        1. Onboarding → 2. IPS Generation → 3. MPT Optimization
        4. Full Stress Testing (all 4 layers)    
    
    Returns:
        List of result dicts, one per investor.
    """
    all_results = []
    total = len(profiles)

    for idx, profile in enumerate(profiles, 1):
        name = profile["name"]
        print(f"\n{'━'*70}")
        print(f"  [{idx}/{total}] Processing: {name}")
        print(f"{'━'*70}")

        result = {
            "profile": profile,
            "name": name,
            "error": None,
            "allocation": None,
            "ips": None,
            "stress_report": None,
        }

        try:
            # Fresh engine per investor to avoid state bleed
            engine = PortfolioBuilder()
            engine.run_api_onboarding(profile)

            # Suppress IPS print output
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()

            with redirect_stdout(f):
                ips_constraints = engine.generate_ips()

            result["ips"] = ips_constraints
            print(f"    IPS: ReturnObj={ips_constraints.get('ReturnObjective',0)*100:.1f}%, "
                  f"Equity=[{ips_constraints.get('EquityMin',0)*100:.0f}-{ips_constraints.get('EquityMax',0)*100:.0f}%], "
                  f"MaxDD={ips_constraints.get('MaxDrawdown',0)*100:.0f}%")

            # Optimization
            allocation = engine.run_mpt_only()
            result["allocation"] = allocation

            top_3 = allocation.drop("CASH_RESERVE", errors='ignore').nlargest(3)
            cash = allocation.get("CASH_RESERVE", 0)
            print(f"    Allocation: Cash={cash*100:.1f}%, Top3={', '.join(f'{t}:{w*100:.1f}%' for t,w in top_3.items())}")

            # Full stress test
            stress_engine = StressTestingEngine(engine)
            with redirect_stdout(io.StringIO()):
                stress_report = stress_engine.run_full_stress_test(allocation, ips_constraints)
            result["stress_report"] = stress_report

            verdict = stress_report.get("final_verdict", {}).get("label", "N/A")
            score = stress_report.get("final_verdict", {}).get("score", 0)
            print(f"    Stress Verdict: {verdict} (Score: {score}/100)")

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            print(f"    ❌ PIPELINE FAILURE: {result['error']}")
            traceback.print_exc()

        all_results.append(result)

    return all_results


# ==============================================================================
# MODULE 3: ATTACK TESTS
# ==============================================================================

# ── Attack 1: Concentration Attack ──

def attack_concentration(results):
    """
    Detects dangerously concentrated portfolios.
    Checks per-asset (>25%), per-sector (>40%), and HHI.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 1: CONCENTRATION ANALYSIS")
    print("="*70)

    findings = []

    for r in results:
        if r["allocation"] is None:
            continue

        name = r["name"]
        alloc = r["allocation"]
        risky = alloc.drop("CASH_RESERVE", errors='ignore')
        risky = risky[risky > 1e-6]

        if risky.sum() < 1e-6:
            findings.append({"name": name, "hhi": 1.0, "flag": "ALL CASH",
                             "max_asset": "N/A", "max_asset_wt": 0,
                             "max_sector": "N/A", "max_sector_wt": 0})
            continue

        normalized = risky / risky.sum()

        # Per-asset check
        max_asset = normalized.idxmax()
        max_asset_wt = normalized.max()
        asset_concentrated = max_asset_wt > 0.25

        # Per-sector check
        sector_weights = {}
        for ticker, wt in normalized.items():
            sector = SECTOR_MAP.get(ticker, "Other")
            sector_weights[sector] = sector_weights.get(sector, 0) + wt

        max_sector = max(sector_weights, key=sector_weights.get)
        max_sector_wt = sector_weights[max_sector]
        sector_concentrated = max_sector_wt > 0.40

        # HHI
        hhi = float(np.sum(normalized.values ** 2))

        if hhi > 0.25:
            flag = "HIGHLY CONCENTRATED"
        elif hhi > 0.15:
            flag = "MODERATELY DIVERSIFIED"
        else:
            flag = "WELL DIVERSIFIED"

        detail = {
            "name": name,
            "hhi": round(hhi, 4),
            "flag": flag,
            "max_asset": max_asset,
            "max_asset_wt": round(float(max_asset_wt) * 100, 2),
            "max_sector": max_sector,
            "max_sector_wt": round(float(max_sector_wt) * 100, 2),
            "asset_breach": asset_concentrated,
            "sector_breach": sector_concentrated,
        }
        findings.append(detail)

        status_icon = "🔴" if flag == "HIGHLY CONCENTRATED" else ("🟡" if flag == "MODERATELY DIVERSIFIED" else "🟢")
        print(f"  {status_icon} {name}: HHI={hhi:.4f} | {flag} | "
              f"Max Asset: {max_asset} ({max_asset_wt*100:.1f}%) | "
              f"Max Sector: {max_sector} ({max_sector_wt*100:.1f}%)")

    return findings


# ── Attack 2: Risk Misalignment Test ──

def attack_risk_misalignment(results):
    """
    Checks if low-risk investors get dangerous drawdowns,
    and if high-risk investors get poor returns.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 2: RISK MISALIGNMENT TEST")
    print("="*70)

    findings = []
    risk_attitudes = []
    expected_returns = []
    max_drawdowns = []

    for r in results:
        if r["stress_report"] is None:
            continue

        name = r["name"]
        risk_att = r["profile"]["risk_attitude"]
        tail = r["stress_report"].get("tail_risk_analysis", {})
        exp_ret = tail.get("expected_return", None)
        max_dd = tail.get("max_drawdown", None)

        if exp_ret is None or max_dd is None:
            continue

        risk_attitudes.append(risk_att)
        expected_returns.append(exp_ret)
        max_drawdowns.append(max_dd)

        flags = []

        # Low risk investor with high drawdown
        if risk_att <= 2 and max_dd > 20:
            flags.append(f"LOW_RISK_HIGH_DD (risk={risk_att}, dd={max_dd:.1f}%)")

        # High risk investor with return below risk-free
        if risk_att >= 4 and exp_ret < RISK_FREE_RATE * 100:
            flags.append(f"HIGH_RISK_LOW_RETURN (risk={risk_att}, ret={exp_ret:.1f}%)")

        status = "FAIL" if flags else "PASS"
        icon = "🔴" if flags else "🟢"

        finding = {
            "name": name,
            "risk_attitude": risk_att,
            "expected_return": round(exp_ret, 2),
            "max_drawdown": round(max_dd, 2),
            "status": status,
            "flags": flags,
        }
        findings.append(finding)

        print(f"  {icon} {name}: Risk={risk_att} | Return={exp_ret:.1f}% | "
              f"MaxDD={max_dd:.1f}% | {status}"
              + (f" → {'; '.join(flags)}" if flags else ""))

    # Compute correlations
    correlations = {}
    if len(risk_attitudes) >= 3:
        risk_arr = np.array(risk_attitudes)
        ret_arr = np.array(expected_returns)
        dd_arr = np.array(max_drawdowns)

        corr_ret = np.corrcoef(risk_arr, ret_arr)[0, 1]
        corr_dd = np.corrcoef(risk_arr, dd_arr)[0, 1]
        correlations = {
            "risk_vs_return": round(float(corr_ret), 4),
            "risk_vs_drawdown": round(float(corr_dd), 4),
        }

        print(f"\n  📊 Correlation: Risk↔Return = {corr_ret:.4f} "
              f"(expect positive, close to +1)")
        print(f"  📊 Correlation: Risk↔Drawdown = {corr_dd:.4f} "
              f"(expect positive, higher risk = higher DD)")

        if corr_ret < 0.3:
            print(f"  🚨 WEAK risk-return correlation! Model may not differentiate risk levels.")
        if corr_dd < 0:
            print(f"  🚨 NEGATIVE risk-drawdown correlation! Model is INVERTED.")

    return {"findings": findings, "correlations": correlations}


# ── Attack 3: Model Differentiation Test ──

def attack_differentiation(results):
    """
    Checks if portfolios for different investor profiles are actually different.
    Uses cosine similarity between weight vectors.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 3: MODEL DIFFERENTIATION TEST")
    print("="*70)

    # Build weight matrix
    valid = [(r["name"], r["allocation"], r["profile"]["risk_attitude"])
             for r in results if r["allocation"] is not None]

    if len(valid) < 2:
        print("  ⚠️  Too few valid portfolios to compare.")
        return {"failures": [], "stats": {}}

    # Align all weight vectors to the same ticker universe
    all_tickers = set()
    for _, alloc, _ in valid:
        risky = alloc.drop("CASH_RESERVE", errors='ignore')
        all_tickers.update(risky.index)

    all_tickers = sorted(all_tickers)

    weight_matrix = []
    names = []
    risk_levels = []
    for name, alloc, risk in valid:
        risky = alloc.drop("CASH_RESERVE", errors='ignore')
        vec = np.array([float(risky.get(t, 0)) for t in all_tickers])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        weight_matrix.append(vec)
        names.append(name)
        risk_levels.append(risk)

    weight_matrix = np.array(weight_matrix)

    # Compute pairwise cosine similarity
    failures = []
    similarities = []

    for (i, j) in combinations(range(len(names)), 2):
        sim = float(np.dot(weight_matrix[i], weight_matrix[j]))
        similarities.append(sim)

        # Flag if similarity > 80% but profiles have DIFFERENT risk levels
        if sim > 0.80 and risk_levels[i] != risk_levels[j]:
            failures.append({
                "pair": f"{names[i]} ↔ {names[j]}",
                "similarity": round(sim * 100, 1),
                "risk_levels": f"{risk_levels[i]} vs {risk_levels[j]}",
            })

    mean_sim = np.mean(similarities) if similarities else 0
    min_sim = np.min(similarities) if similarities else 0
    max_sim = np.max(similarities) if similarities else 0

    print(f"  📊 Pairwise Similarity — Mean: {mean_sim*100:.1f}% | "
          f"Min: {min_sim*100:.1f}% | Max: {max_sim*100:.1f}%")

    if failures:
        print(f"\n  🚨 {len(failures)} DIFFERENTIATION FAILURES (>80% similar, different risk):")
        for f in failures:
            print(f"     → {f['pair']} = {f['similarity']}% similar (Risk: {f['risk_levels']})")
    else:
        print(f"\n  ✅ All portfolios with different risk levels show adequate differentiation.")

    stats = {
        "mean_similarity": round(mean_sim * 100, 2),
        "min_similarity": round(min_sim * 100, 2),
        "max_similarity": round(max_sim * 100, 2),
        "total_pairs": len(similarities),
        "failure_count": len(failures),
    }

    return {"failures": failures, "stats": stats}


# ── Attack 4: Sensitivity Attack ──

def attack_sensitivity(results):
    """
    Perturbs risk_attitude ±1 for select profiles and measures allocation shift.
    Flags UNSTABLE (>50% turnover) or OVER-RIGID (<2% turnover).
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 4: SENSITIVITY ATTACK")
    print("="*70)

    # Select 3 profiles: lowest risk, mid risk, highest risk
    valid = [r for r in results if r["allocation"] is not None and r["ips"] is not None]
    if not valid:
        print("  ⚠️  No valid results to test.")
        return []

    sorted_by_risk = sorted(valid, key=lambda x: x["profile"]["risk_attitude"])
    candidates = [sorted_by_risk[0], sorted_by_risk[len(sorted_by_risk)//2], sorted_by_risk[-1]]

    findings = []

    for r in candidates:
        name = r["name"]
        profile = r["profile"]
        base_alloc = r["allocation"]
        base_risky = base_alloc.drop("CASH_RESERVE", errors='ignore')

        for delta in [-1, +1]:
            perturbed_risk = max(1, min(5, profile["risk_attitude"] + delta))
            if perturbed_risk == profile["risk_attitude"]:
                continue

            perturbed_profile = copy.deepcopy(profile)
            perturbed_profile["risk_attitude"] = perturbed_risk
            perturbed_profile["name"] = f"{name}_risk{'+' if delta > 0 else ''}{delta}"

            try:
                import io
                from contextlib import redirect_stdout

                engine = PortfolioBuilder()
                engine.run_api_onboarding(perturbed_profile)
                with redirect_stdout(io.StringIO()):
                    ips = engine.generate_ips()
                perturbed_alloc = engine.run_mpt_only()
                perturbed_risky = perturbed_alloc.drop("CASH_RESERVE", errors='ignore')

                # Align
                all_t = base_risky.index.union(perturbed_risky.index)
                bw = base_risky.reindex(all_t, fill_value=0)
                pw = perturbed_risky.reindex(all_t, fill_value=0)

                diff = (pw - bw).abs()
                turnover = float(diff.sum()) / 2

                if turnover > 0.50:
                    flag = "UNSTABLE"
                elif turnover < 0.02:
                    flag = "OVER-RIGID"
                else:
                    flag = "NORMAL"

                icon = "🔴" if flag != "NORMAL" else "🟢"
                print(f"  {icon} {name} (risk {profile['risk_attitude']}→{perturbed_risk}): "
                      f"Turnover={turnover*100:.1f}% → {flag}")

                findings.append({
                    "name": name,
                    "base_risk": profile["risk_attitude"],
                    "perturbed_risk": perturbed_risk,
                    "turnover_pct": round(turnover * 100, 2),
                    "flag": flag,
                })

            except Exception as e:
                print(f"  ❌ {name} (risk {profile['risk_attitude']}→{perturbed_risk}): FAILED — {e}")
                findings.append({
                    "name": name,
                    "base_risk": profile["risk_attitude"],
                    "perturbed_risk": perturbed_risk,
                    "turnover_pct": None,
                    "flag": "ERROR",
                })

    return findings


# ── Attack 5: Extreme Market Shock Test ──

def attack_extreme_shocks(results):
    """
    Applies 5 extreme shock scenarios beyond the standard stress tests.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 5: EXTREME MARKET SHOCK TEST")
    print("="*70)

    EXTREME_SHOCKS = {
        "Global Crash (-40% uniform)": {t: -0.40 for t in SECTOR_MAP.values()},
        "IT Sector Collapse (-60%)": {"IT": -0.60, "Pharma": -0.05, "_default": -0.10},
        "Rate Shock (+300bps)": {"IT": -0.20, "Pharma": -0.10, "Banking": -0.30, "_default": -0.15},
        "Correlation→1 Crisis": {t: -0.35 for t in set(SECTOR_MAP.values())},
        "Combined Armageddon": {t: -0.55 for t in set(SECTOR_MAP.values())},
    }

    findings = []

    for r in results:
        if r["allocation"] is None:
            continue

        name = r["name"]
        alloc = r["allocation"]
        risky = alloc.drop("CASH_RESERVE", errors='ignore')
        risky = risky[risky > 1e-6]
        ips = r.get("ips", {})
        max_dd = ips.get("MaxDrawdown", 0.15)

        for shock_name, shocks in EXTREME_SHOCKS.items():
            port_loss = 0.0
            for ticker, wt in risky.items():
                sector = SECTOR_MAP.get(ticker, "Other")
                shock = shocks.get(sector, shocks.get("_default", -0.20))
                port_loss += shock * wt

            drawdown = abs(min(port_loss, 0))
            ips_breached = drawdown > max_dd

            if drawdown > max_dd * 1.5:
                status = "CRITICAL FAILURE"
            elif ips_breached:
                status = "WARNING"
            else:
                status = "SAFE"

            findings.append({
                "name": name,
                "shock": shock_name,
                "portfolio_loss_pct": round(port_loss * 100, 2),
                "drawdown_pct": round(drawdown * 100, 2),
                "max_dd_threshold": round(max_dd * 100, 2),
                "ips_breached": ips_breached,
                "status": status,
            })

    # Summary
    critical = sum(1 for f in findings if f["status"] == "CRITICAL FAILURE")
    warning = sum(1 for f in findings if f["status"] == "WARNING")
    safe = sum(1 for f in findings if f["status"] == "SAFE")

    print(f"  📊 Results: {safe} SAFE | {warning} WARNING | {critical} CRITICAL")
    print(f"  Total shock-profile combinations tested: {len(findings)}")

    return findings


# ── Attack 6: IPS Violation Test (aggregate) ──

def attack_ips_violations(results):
    """
    Aggregates IPS violation results from stress testing across all profiles.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 6: IPS VIOLATION AGGREGATION")
    print("="*70)

    findings = []

    for r in results:
        if r["stress_report"] is None:
            continue

        name = r["name"]
        ips_val = r["stress_report"].get("ips_validation", {})
        compliance = ips_val.get("compliance_score", 0)
        safe = ips_val.get("safe", 0)
        warning = ips_val.get("warning", 0)
        critical = ips_val.get("critical", 0)

        if compliance >= 75:
            status = "SAFE"
            icon = "🟢"
        elif compliance >= 40:
            status = "WARNING"
            icon = "🟡"
        else:
            status = "CRITICAL FAILURE"
            icon = "🔴"

        finding = {
            "name": name,
            "compliance_score": compliance,
            "safe_scenarios": safe,
            "warning_scenarios": warning,
            "critical_scenarios": critical,
            "status": status,
        }
        findings.append(finding)
        print(f"  {icon} {name}: Compliance={compliance:.0f}% | "
              f"Safe={safe} Warning={warning} Critical={critical}")

    return findings


# ── Attack 7: Tail Risk Analysis ──

def attack_tail_risk(results):
    """
    Checks if tail risk is proportionate to investor risk profile.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 7: TAIL RISK PROPORTIONALITY")
    print("="*70)

    findings = []

    for r in results:
        if r["stress_report"] is None:
            continue

        name = r["name"]
        risk_att = r["profile"]["risk_attitude"]
        tail = r["stress_report"].get("tail_risk_analysis", {})

        if "error" in tail:
            continue

        var_95 = tail.get("var_95", 0)
        cvar_95 = tail.get("cvar_95", 0)
        black_swan = tail.get("black_swan_loss", 0)

        flags = []

        # Conservative with excessive CVaR
        if risk_att <= 2 and cvar_95 > 25:
            flags.append(f"EXCESSIVE_TAIL_RISK (CVaR={cvar_95:.1f}% for risk={risk_att})")

        # Aggressive with suspiciously low CVaR
        if risk_att >= 4 and cvar_95 < 10:
            flags.append(f"SUSPICIOUSLY_LOW_TAIL (CVaR={cvar_95:.1f}% for risk={risk_att})")

        status = "FLAG" if flags else "PASS"
        icon = "🔴" if flags else "🟢"

        finding = {
            "name": name,
            "risk_attitude": risk_att,
            "var_95": round(var_95, 2),
            "cvar_95": round(cvar_95, 2),
            "black_swan_loss": round(black_swan, 2),
            "status": status,
            "flags": flags,
        }
        findings.append(finding)
        print(f"  {icon} {name}: Risk={risk_att} | VaR={var_95:.1f}% | "
              f"CVaR={cvar_95:.1f}% | BlackSwan={black_swan:.1f}% | {status}")

    return findings


# ── Attack 8: Optimizer Failure Detection ──

def attack_optimizer_failures(results):
    """
    Checks for optimizer collapse: extreme weights, constraint violations.
    """
    print("\n" + "="*70)
    print("  🔴 ATTACK 8: OPTIMIZER FAILURE DETECTION")
    print("="*70)

    findings = []

    for r in results:
        if r["allocation"] is None:
            continue

        name = r["name"]
        alloc = r["allocation"]
        ips = r.get("ips", {})
        risky = alloc.drop("CASH_RESERVE", errors='ignore')
        cash = alloc.get("CASH_RESERVE", 0)

        flags = []

        # Single weight > 50%
        if risky.max() > 0.50:
            top = risky.idxmax()
            flags.append(f"COLLAPSED_TO_{top} ({risky.max()*100:.1f}%)")

        # Top 3 > 70%
        top3_sum = risky.nlargest(3).sum()
        if top3_sum > 0.70:
            flags.append(f"OVER_CONCENTRATED_TOP3 ({top3_sum*100:.1f}%)")

        # Negative weights
        if (risky < -1e-6).any():
            neg_tickers = risky[risky < -1e-6].index.tolist()
            flags.append(f"NEGATIVE_WEIGHTS ({neg_tickers})")

        # Cash breach
        cash_min = ips.get("CashMin", 0)
        if cash < cash_min - 0.01:  # 1% tolerance
            flags.append(f"CASH_BREACH (actual={cash*100:.1f}%, min={cash_min*100:.1f}%)")

        # Equity bounds breach
        eq_total = risky.sum()
        eq_min = ips.get("EquityMin", 0)
        eq_max = ips.get("EquityMax", 1)
        if eq_total < eq_min - 0.01:
            flags.append(f"EQUITY_BELOW_MIN (actual={eq_total*100:.1f}%, min={eq_min*100:.1f}%)")
        if eq_total > eq_max + 0.01:
            flags.append(f"EQUITY_ABOVE_MAX (actual={eq_total*100:.1f}%, max={eq_max*100:.1f}%)")

        status = "FAIL" if flags else "PASS"
        icon = "🔴" if flags else "🟢"

        finding = {
            "name": name,
            "cash_pct": round(float(cash) * 100, 2),
            "equity_pct": round(float(eq_total) * 100, 2),
            "max_single_weight": round(float(risky.max()) * 100, 2),
            "top3_weight": round(float(top3_sum) * 100, 2),
            "status": status,
            "flags": flags,
        }
        findings.append(finding)
        print(f"  {icon} {name}: Cash={cash*100:.1f}% | Equity={eq_total*100:.1f}% | "
              f"MaxWt={risky.max()*100:.1f}% | Top3={top3_sum*100:.1f}% | {status}"
              + (f"\n     → {'; '.join(flags)}" if flags else ""))

    return findings


# ==============================================================================
# MODULE 4: RESULTS AGGREGATION & CSV EXPORT
# ==============================================================================

def build_master_dataframe(results, concentration, risk_alignment, differentiation,
                           sensitivity, extreme_shocks, ips_violations,
                           tail_risk_findings, optimizer_findings):
    """
    Builds the master audit DataFrame with one row per investor.
    """
    rows = []

    for r in results:
        if r["allocation"] is None:
            continue

        name = r["name"]
        profile = r["profile"]
        alloc = r["allocation"]
        ips = r.get("ips", {})
        stress = r.get("stress_report", {})
        tail = stress.get("tail_risk_analysis", {}) if stress else {}
        sensitivity_data = stress.get("sensitivity_analysis", {}) if stress else {}
        verdict = stress.get("final_verdict", {}) if stress else {}

        risky = alloc.drop("CASH_RESERVE", errors='ignore')

        # Find concentration finding
        conc_f = next((c for c in concentration if c["name"] == name), {})
        # Risk alignment finding
        risk_f = next((f for f in risk_alignment.get("findings", []) if f["name"] == name), {})
        # IPS finding
        ips_f = next((f for f in ips_violations if f["name"] == name), {})
        # Tail risk finding
        tail_f = next((f for f in tail_risk_findings if f["name"] == name), {})
        # Optimizer finding
        opt_f = next((f for f in optimizer_findings if f["name"] == name), {})

        row = {
            "investor_name": name,
            "age": profile.get("age"),
            "risk_attitude": profile.get("risk_attitude"),
            "goal": profile.get("goal"),
            "goal_years": profile.get("goal_years"),
            "capital": profile.get("capital"),
            "liquidity_pct": profile.get("liquidity"),
            "loss_tolerance_pct": profile.get("loss_tolerance"),

            # IPS
            "ips_return_obj": round(ips.get("ReturnObjective", 0) * 100, 2),
            "ips_equity_min": round(ips.get("EquityMin", 0) * 100, 2),
            "ips_equity_max": round(ips.get("EquityMax", 0) * 100, 2),
            "ips_max_drawdown": round(ips.get("MaxDrawdown", 0) * 100, 2),

            # Portfolio
            "cash_pct": round(float(alloc.get("CASH_RESERVE", 0)) * 100, 2),
            "equity_pct": round(float(risky.sum()) * 100, 2),
            "num_holdings": int((risky > 0.001).sum()),
            "max_single_weight": round(float(risky.max()) * 100, 2),

            # Risk Metrics
            "expected_return": tail.get("expected_return", None),
            "volatility": tail.get("portfolio_volatility", None),
            "var_95": tail.get("var_95", None),
            "cvar_95": tail.get("cvar_95", None),
            "max_drawdown_sim": tail.get("max_drawdown", None),
            "black_swan_loss": tail.get("black_swan_loss", None),

            # Stability
            "stability_score": sensitivity_data.get("stability_score", None),

            # Concentration
            "hhi": conc_f.get("hhi", None),
            "concentration_flag": conc_f.get("flag", None),

            # IPS Compliance
            "ips_compliance_pct": ips_f.get("compliance_score", None),

            # Attack Flags
            "risk_alignment": risk_f.get("status", None),
            "optimizer_status": opt_f.get("status", None),
            "tail_risk_status": tail_f.get("status", None),

            # Verdict
            "stress_verdict": verdict.get("label", None),
            "stress_score": verdict.get("score", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ==============================================================================
# MODULE 5: SYSTEM-LEVEL DIAGNOSTICS
# ==============================================================================

def compute_system_diagnostics(df, differentiation, risk_alignment):
    """
    Computes aggregate system-level diagnostic metrics.
    """
    print("\n" + "="*70)
    print("  📊 SYSTEM-LEVEL DIAGNOSTICS")
    print("="*70)

    total = len(df)

    diagnostics = {}

    # % concentrated
    concentrated = (df["concentration_flag"] == "HIGHLY CONCENTRATED").sum()
    diagnostics["pct_concentrated"] = round(concentrated / total * 100, 1) if total > 0 else 0
    print(f"  • Concentrated Portfolios: {concentrated}/{total} ({diagnostics['pct_concentrated']:.0f}%)")

    # % IPS failures
    ips_fail = (df["ips_compliance_pct"] < 50).sum() if "ips_compliance_pct" in df.columns else 0
    diagnostics["pct_ips_failures"] = round(ips_fail / total * 100, 1) if total > 0 else 0
    print(f"  • IPS Compliance < 50%: {ips_fail}/{total} ({diagnostics['pct_ips_failures']:.0f}%)")

    # Risk-return correlation
    corr = risk_alignment.get("correlations", {})
    diagnostics["risk_return_correlation"] = corr.get("risk_vs_return", None)
    diagnostics["risk_drawdown_correlation"] = corr.get("risk_vs_drawdown", None)
    print(f"  • Risk↔Return Correlation: {diagnostics['risk_return_correlation']}")
    print(f"  • Risk↔Drawdown Correlation: {diagnostics['risk_drawdown_correlation']}")

    # Avg drawdown per risk level
    if "risk_attitude" in df.columns and "max_drawdown_sim" in df.columns:
        dd_by_risk = df.groupby("risk_attitude")["max_drawdown_sim"].mean()
        diagnostics["avg_drawdown_by_risk"] = dd_by_risk.to_dict()
        print(f"  • Avg Drawdown by Risk Level:")
        for risk_level, dd in sorted(dd_by_risk.items()):
            if pd.notna(dd):
                print(f"      Risk {risk_level}: {dd:.1f}%")

    # Mean portfolio similarity
    diff_stats = differentiation.get("stats", {})
    diagnostics["mean_portfolio_similarity"] = diff_stats.get("mean_similarity", None)
    diagnostics["differentiation_failures"] = diff_stats.get("failure_count", 0)
    print(f"  • Mean Portfolio Similarity: {diagnostics['mean_portfolio_similarity']}%")
    print(f"  • Differentiation Failures: {diagnostics['differentiation_failures']}")

    # Optimizer failures
    opt_fail = (df["optimizer_status"] == "FAIL").sum() if "optimizer_status" in df.columns else 0
    diagnostics["pct_optimizer_failures"] = round(opt_fail / total * 100, 1) if total > 0 else 0
    print(f"  • Optimizer Failures: {opt_fail}/{total} ({diagnostics['pct_optimizer_failures']:.0f}%)")

    return diagnostics


# ==============================================================================
# MODULE 6: FINAL SYSTEM VERDICT
# ==============================================================================

def compute_system_verdict(df, concentration, risk_alignment, differentiation,
                           sensitivity, ips_violations, tail_risk_findings,
                           optimizer_findings, diagnostics):
    """
    Computes the aggregate system-level ROBUST / MODERATE / FRAGILE verdict.
    """
    print("\n" + "="*70)
    print("  🏛️  FINAL SYSTEM VERDICT")
    print("="*70)

    score = 100
    deductions = []
    total = len(df)

    # Concentration deductions
    concentrated = sum(1 for c in concentration if c.get("flag") == "HIGHLY CONCENTRATED")
    if concentrated > 0:
        d = concentrated * 3
        score -= d
        deductions.append(f"-{d} pts: {concentrated} highly concentrated portfolios")

    # Risk misalignment deductions
    misaligned = sum(1 for f in risk_alignment.get("findings", []) if f["status"] == "FAIL")
    if misaligned > 0:
        d = misaligned * 5
        score -= d
        deductions.append(f"-{d} pts: {misaligned} risk-misalignment failures")

    # Differentiation deductions
    diff_failures = differentiation.get("stats", {}).get("failure_count", 0)
    if diff_failures > 0:
        d = diff_failures * 8
        score -= d
        deductions.append(f"-{d} pts: {diff_failures} differentiation failures")

    # Sensitivity deductions
    unstable = sum(1 for s in sensitivity if s.get("flag") == "UNSTABLE")
    rigid = sum(1 for s in sensitivity if s.get("flag") == "OVER-RIGID")
    if unstable > 0:
        d = unstable * 5
        score -= d
        deductions.append(f"-{d} pts: {unstable} sensitivity instabilities")
    if rigid > 0:
        d = rigid * 3
        score -= d
        deductions.append(f"-{d} pts: {rigid} over-rigid responses")

    # IPS critical failures
    ips_critical = sum(1 for f in ips_violations if f.get("status") == "CRITICAL FAILURE")
    if ips_critical > 0:
        d = ips_critical * 5
        score -= d
        deductions.append(f"-{d} pts: {ips_critical} IPS critical failures")

    # Optimizer collapses
    opt_fail = sum(1 for f in optimizer_findings if f.get("status") == "FAIL")
    if opt_fail > 0:
        d = opt_fail * 10
        score -= d
        deductions.append(f"-{d} pts: {opt_fail} optimizer failures")

    # Tail risk deductions
    tail_flags = sum(1 for f in tail_risk_findings if f.get("status") == "FLAG")
    if tail_flags > 0:
        d = tail_flags * 3
        score -= d
        deductions.append(f"-{d} pts: {tail_flags} tail risk disproportionalities")

    # Weak correlation deduction
    corr_ret = diagnostics.get("risk_return_correlation")
    if corr_ret is not None and corr_ret < 0.3:
        score -= 10
        deductions.append(f"-10 pts: Weak risk-return correlation ({corr_ret:.3f})")

    score = max(0, min(100, score))

    if score >= 70:
        label = "ROBUST SYSTEM"
        emoji = "✅"
    elif score >= 40:
        label = "MODERATE — NEEDS IMPROVEMENT"
        emoji = "⚠️"
    else:
        label = "FRAGILE — STRUCTURAL FAILURES"
        emoji = "🚨"

    print(f"\n  {'='*60}")
    print(f"  {emoji}  VERDICT: {label}")
    print(f"  {'='*60}")
    print(f"  System Robustness Score: {score}/100")
    print(f"\n  Scoring Breakdown:")
    print(f"    Starting score: 100")
    for d in deductions:
        print(f"    {d}")
    print(f"    ────────────────────────")
    print(f"    Final Score: {score}/100")

    return {
        "label": label,
        "score": score,
        "deductions": deductions,
    }


# ==============================================================================
# MODULE 7: VISUALIZATIONS
# ==============================================================================

def generate_all_visualizations(df, results, concentration, save_dir,
                                 differentiation=None, sensitivity=None):
    """
    Generates 5 publication-quality charts.
    """
    print(f"\n  📈 Generating visualizations to {save_dir}/...")
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        'figure.facecolor': THEME['bg'],
        'axes.facecolor': THEME['bg'],
        'axes.edgecolor': THEME['grid'],
        'axes.labelcolor': THEME['text'],
        'text.color': THEME['text'],
        'xtick.color': THEME['text'],
        'ytick.color': THEME['text'],
        'grid.color': THEME['grid'],
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    # ── CHART 1: Risk vs Return Scatter ──
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    valid_df = df.dropna(subset=["expected_return", "volatility", "risk_attitude"])
    if not valid_df.empty:
        risk_colors = {1: THEME['blue'], 2: THEME['green'], 3: THEME['yellow'],
                       4: THEME['orange'], 5: THEME['red']}
        colors = [risk_colors.get(int(r), THEME['text']) for r in valid_df["risk_attitude"]]

        scatter = ax1.scatter(
            valid_df["volatility"], valid_df["expected_return"],
            c=colors, s=180, alpha=0.85, edgecolors='white', linewidths=0.5, zorder=5
        )

        for _, row in valid_df.iterrows():
            short_name = row["investor_name"][:18]
            ax1.annotate(
                short_name, (row["volatility"], row["expected_return"]),
                xytext=(8, 8), textcoords='offset points', fontsize=7,
                color=THEME['text'], alpha=0.8
            )

        # Risk-free rate line
        ax1.axhline(y=RISK_FREE_RATE * 100, color=THEME['purple'], linewidth=1,
                     linestyle='--', alpha=0.6, label=f'Risk-Free Rate ({RISK_FREE_RATE*100:.1f}%)')

        legend_patches = [Patch(facecolor=risk_colors[i], label=f'Risk {i}') for i in range(1, 6)]
        ax1.legend(handles=legend_patches, loc='upper left', fontsize=9,
                   facecolor=THEME['bg'], edgecolor=THEME['grid'])

    ax1.set_xlabel("Portfolio Volatility (%)", fontsize=12)
    ax1.set_ylabel("Expected Annual Return (%)", fontsize=12)
    ax1.set_title("RISK vs RETURN — ALL INVESTOR PROFILES", fontsize=15,
                   fontweight='bold', pad=15, color=THEME['blue'])
    ax1.grid(True, alpha=0.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "01_risk_vs_return.png"), dpi=150,
                 bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig1)
    print("    ✓ 01_risk_vs_return.png")

    # ── CHART 2: Drawdown Comparison ──
    fig2, ax2 = plt.subplots(figsize=(14, 9))

    dd_df = df.dropna(subset=["max_drawdown_sim"]).sort_values("max_drawdown_sim", ascending=True)
    if not dd_df.empty:
        risk_colors_map = {1: THEME['blue'], 2: THEME['green'], 3: THEME['yellow'],
                           4: THEME['orange'], 5: THEME['red']}
        bar_colors = [risk_colors_map.get(int(r), THEME['text']) for r in dd_df["risk_attitude"]]

        short_names = [n[:22] for n in dd_df["investor_name"]]
        bars = ax2.barh(range(len(short_names)), dd_df["max_drawdown_sim"],
                        color=bar_colors, edgecolor='none', height=0.65, alpha=0.9)

        for bar, val in zip(bars, dd_df["max_drawdown_sim"]):
            ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va='center', fontsize=8, color=THEME['text'])

        ax2.set_yticks(range(len(short_names)))
        ax2.set_yticklabels(short_names, fontsize=8)

    ax2.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax2.set_title("MAX DRAWDOWN COMPARISON — BY INVESTOR", fontsize=15,
                   fontweight='bold', pad=15, color=THEME['blue'])
    ax2.grid(axis='x', alpha=0.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "02_drawdown_comparison.png"), dpi=150,
                 bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig2)
    print("    ✓ 02_drawdown_comparison.png")

    # ── CHART 3: Allocation Heatmap ──
    fig3, ax3 = plt.subplots(figsize=(16, 10))

    # Build allocation matrix
    alloc_data = {}
    for r in results:
        if r["allocation"] is not None:
            risky = r["allocation"].drop("CASH_RESERVE", errors='ignore')
            alloc_data[r["name"][:20]] = (risky * 100).to_dict()

    if alloc_data:
        alloc_df = pd.DataFrame(alloc_data).T.fillna(0)
        # Only show tickers with at least one material allocation
        material = alloc_df.columns[alloc_df.max() > 0.5]
        if len(material) > 0:
            alloc_df = alloc_df[material]

        cmap = LinearSegmentedColormap.from_list(
            'alloc_heat',
            [(0, THEME['bg']), (0.3, THEME['blue']),
             (0.6, THEME['green']), (1.0, THEME['yellow'])]
        )

        im = ax3.imshow(alloc_df.values, cmap=cmap, aspect='auto', vmin=0)

        ax3.set_xticks(range(len(alloc_df.columns)))
        ax3.set_xticklabels(alloc_df.columns, fontsize=8, rotation=45, ha='right')
        ax3.set_yticks(range(len(alloc_df.index)))
        ax3.set_yticklabels(alloc_df.index, fontsize=7)

        # Annotate
        for i in range(alloc_df.shape[0]):
            for j in range(alloc_df.shape[1]):
                val = alloc_df.iloc[i, j]
                if val > 0.5:
                    color = 'black' if val > 15 else THEME['text']
                    ax3.text(j, i, f"{val:.0f}", ha='center', va='center',
                             fontsize=6, color=color, fontweight='bold')

        cbar = fig3.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
        cbar.set_label("Allocation Weight (%)", fontsize=10, color=THEME['text'])
        cbar.ax.yaxis.set_tick_params(color=THEME['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=THEME['text'])

    ax3.set_title("PORTFOLIO ALLOCATION HEATMAP — ALL INVESTORS", fontsize=15,
                   fontweight='bold', pad=15, color=THEME['blue'])
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "03_allocation_heatmap.png"), dpi=150,
                 bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig3)
    print("    ✓ 03_allocation_heatmap.png")

    # ── CHART 4: Concentration Distribution ──
    fig4, ax4 = plt.subplots(figsize=(12, 7))

    hhi_values = [c["hhi"] for c in concentration if c.get("hhi") is not None]
    if hhi_values:
        ax4.hist(hhi_values, bins=15, color=THEME['blue'], alpha=0.8,
                 edgecolor=THEME['grid'], linewidth=0.5)

        ax4.axvline(x=0.25, color=THEME['red'], linewidth=2, linestyle='--',
                    label='High Concentration (HHI=0.25)', alpha=0.8)
        ax4.axvline(x=0.15, color=THEME['yellow'], linewidth=2, linestyle='--',
                    label='Moderate (HHI=0.15)', alpha=0.8)

        ax4.legend(fontsize=10, facecolor=THEME['bg'], edgecolor=THEME['grid'])

    ax4.set_xlabel("Herfindahl-Hirschman Index (HHI)", fontsize=12)
    ax4.set_ylabel("Number of Portfolios", fontsize=12)
    ax4.set_title("PORTFOLIO CONCENTRATION DISTRIBUTION", fontsize=15,
                   fontweight='bold', pad=15, color=THEME['blue'])
    ax4.grid(axis='y', alpha=0.15)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, "04_concentration_dist.png"), dpi=150,
                 bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig4)
    print("    ✓ 04_concentration_dist.png")

    # ── CHART 5: Failure Frequency Chart ──
    fig5, ax5 = plt.subplots(figsize=(14, 7))

    failure_types = [
        "Concentration",
        "Risk Misalignment",
        "Differentiation",
        "Sensitivity",
        "IPS Violation",
        "Tail Risk",
        "Optimizer Failure",
    ]

    # Count failures
    concentrated_count = sum(1 for c in concentration if c.get("flag") == "HIGHLY CONCENTRATED")
    from_risk = sum(1 for f in df.itertuples() if getattr(f, "risk_alignment", None) == "FAIL") if "risk_alignment" in df.columns else 0
    from_diff = differentiation.get("stats", {}).get("failure_count", 0) if differentiation else 0
    from_sens = sum(1 for s in sensitivity if s.get("flag") in ["UNSTABLE", "OVER-RIGID"]) if sensitivity else 0
    from_ips = sum(1 for f in df.itertuples() if getattr(f, "ips_compliance_pct", 100) < 50) if "ips_compliance_pct" in df.columns else 0
    from_tail = sum(1 for f in df.itertuples() if getattr(f, "tail_risk_status", None) == "FLAG") if "tail_risk_status" in df.columns else 0
    from_opt = sum(1 for f in df.itertuples() if getattr(f, "optimizer_status", None) == "FAIL") if "optimizer_status" in df.columns else 0

    counts = [concentrated_count, from_risk, from_diff, from_sens, from_ips, from_tail, from_opt]
    bar_colors = [THEME['red'] if c > 0 else THEME['green'] for c in counts]

    bars = ax5.bar(range(len(failure_types)), counts, color=bar_colors,
                   edgecolor='none', width=0.6, alpha=0.9)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     str(count), ha='center', fontsize=11, fontweight='bold',
                     color=THEME['red'])

    ax5.set_xticks(range(len(failure_types)))
    ax5.set_xticklabels(failure_types, fontsize=9, rotation=20, ha='right')
    ax5.set_ylabel("Failure Count", fontsize=12)
    ax5.set_title("FAILURE FREQUENCY ACROSS ALL ATTACK TESTS", fontsize=15,
                   fontweight='bold', pad=15, color=THEME['blue'])
    ax5.grid(axis='y', alpha=0.15)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    fig5.tight_layout()
    fig5.savefig(os.path.join(save_dir, "05_failure_frequency.png"), dpi=150,
                 bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig5)
    print("    ✓ 05_failure_frequency.png")

    print(f"  ✅ All charts saved to {save_dir}/")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    """
    Master orchestrator for the full system validation.
    """
    start_time = time.time()

    print("\n" + "█" * 72)
    print("█" + " " * 70 + "█")
    print("█   AssetOS — INSTITUTIONAL-GRADE SYSTEM VALIDATION ENGINE        █")
    print("█   Black-Box Adversarial Testing Suite                           █")
    print("█" + " " * 70 + "█")
    print("█" * 72)
    print(f"\n  Profiles to test: {len(EXTREME_PROFILES)}")
    print(f"  Attack vectors: 8")
    print(f"  Output directory: {OUTPUT_DIR}/")

    # ── STEP 1: Run Full Pipeline ──
    print("\n" + "▓" * 72)
    print("  STEP 1/5: RUNNING FULL PIPELINE FOR ALL PROFILES")
    print("▓" * 72)

    all_results = run_full_pipeline(EXTREME_PROFILES)

    successful = sum(1 for r in all_results if r["allocation"] is not None)
    failed = sum(1 for r in all_results if r["allocation"] is None)
    print(f"\n  Pipeline Summary: {successful} successful, {failed} failed out of {len(EXTREME_PROFILES)}")

    # ── STEP 2: Run All 8 Attacks ──
    print("\n" + "▓" * 72)
    print("  STEP 2/5: EXECUTING 8 ATTACK VECTORS")
    print("▓" * 72)

    concentration_result = attack_concentration(all_results)
    risk_alignment_result = attack_risk_misalignment(all_results)
    differentiation_result = attack_differentiation(all_results)
    sensitivity_result = attack_sensitivity(all_results)
    extreme_shocks_result = attack_extreme_shocks(all_results)
    ips_violations_result = attack_ips_violations(all_results)
    tail_risk_result = attack_tail_risk(all_results)
    optimizer_result = attack_optimizer_failures(all_results)

    # ── STEP 3: Build Master DataFrame ──
    print("\n" + "▓" * 72)
    print("  STEP 3/5: AGGREGATING RESULTS & EXPORTING CSV")
    print("▓" * 72)

    master_df = build_master_dataframe(
        all_results, concentration_result, risk_alignment_result,
        differentiation_result, sensitivity_result, extreme_shocks_result,
        ips_violations_result, tail_risk_result, optimizer_result
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    master_df.to_csv(csv_path, index=False)
    print(f"  ✅ Master audit CSV saved: {csv_path}")
    print(f"  📊 Shape: {master_df.shape[0]} investors × {master_df.shape[1]} metrics")

    # Print summary table
    print("\n  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("  │                              INVESTOR AUDIT SUMMARY TABLE                                             │")
    print("  ├──────────────────────────┬──────┬────────┬────────┬────────┬──────────┬──────────┬─────────┬───────────┤")
    print("  │ Investor                 │ Risk │ Ret %  │ Vol %  │ DD %   │ VaR95 %  │ HHI      │ IPS %   │ Verdict   │")
    print("  ├──────────────────────────┼──────┼────────┼────────┼────────┼──────────┼──────────┼─────────┼───────────┤")

    for _, row in master_df.iterrows():
        name = str(row.get("investor_name", ""))[:24].ljust(24)
        risk = str(int(row.get("risk_attitude", 0))).center(4)
        ret = f"{row.get('expected_return', 0):.1f}".rjust(6) if pd.notna(row.get("expected_return")) else "  N/A ".rjust(6)
        vol = f"{row.get('volatility', 0):.1f}".rjust(6) if pd.notna(row.get("volatility")) else "  N/A ".rjust(6)
        dd = f"{row.get('max_drawdown_sim', 0):.1f}".rjust(6) if pd.notna(row.get("max_drawdown_sim")) else "  N/A ".rjust(6)
        var = f"{row.get('var_95', 0):.1f}".rjust(8) if pd.notna(row.get("var_95")) else "    N/A ".rjust(8)
        hhi = f"{row.get('hhi', 0):.4f}".rjust(8) if pd.notna(row.get("hhi")) else "    N/A ".rjust(8)
        ips = f"{row.get('ips_compliance_pct', 0):.0f}".rjust(7) if pd.notna(row.get("ips_compliance_pct")) else "   N/A ".rjust(7)
        verdict = str(row.get("stress_verdict", "N/A"))[:9].center(9)

        print(f"  │ {name} │ {risk} │ {ret} │ {vol} │ {dd} │ {var} │ {hhi} │ {ips} │ {verdict} │")

    print("  └──────────────────────────┴──────┴────────┴────────┴────────┴──────────┴──────────┴─────────┴───────────┘")

    # ── STEP 4: System Diagnostics & Verdict ──
    print("\n" + "▓" * 72)
    print("  STEP 4/5: SYSTEM-LEVEL DIAGNOSTICS & VERDICT")
    print("▓" * 72)

    diagnostics = compute_system_diagnostics(master_df, differentiation_result, risk_alignment_result)

    verdict = compute_system_verdict(
        master_df, concentration_result, risk_alignment_result,
        differentiation_result, sensitivity_result, ips_violations_result,
        tail_risk_result, optimizer_result, diagnostics
    )

    # ── STEP 5: Visualizations ──
    print("\n" + "▓" * 72)
    print("  STEP 5/5: GENERATING VISUALIZATIONS")
    print("▓" * 72)

    generate_all_visualizations(
        master_df, all_results, concentration_result, OUTPUT_DIR,
        differentiation=differentiation_result,
        sensitivity=sensitivity_result
    )

    # ── FINAL SUMMARY ──
    elapsed = time.time() - start_time

    print("\n" + "█" * 72)
    print("█" + " " * 70 + "█")
    print(f"█   VALIDATION COMPLETE                                            █")
    print(f"█   Time Elapsed: {elapsed:.1f}s                                          █")
    print(f"█   Profiles Tested: {successful}/{len(EXTREME_PROFILES)}                                          █")
    print(f"█   Attack Vectors: 8                                              █")
    print(f"█   System Verdict: {verdict['label'][:40]:<40s}     █")
    print(f"█   Robustness Score: {verdict['score']}/100                                      █")
    print("█" + " " * 70 + "█")
    print("█" * 72)
    print(f"\n  📁 Results exported to: {OUTPUT_DIR}/")
    print(f"  📊 CSV: {os.path.join(OUTPUT_DIR, CSV_FILENAME)}")
    print(f"  📈 Charts: {OUTPUT_DIR}/*.png")
    print()


if __name__ == "__main__":
    main()
