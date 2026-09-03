import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependencies
# We wrap these because not every environment has yfinance/scipy installed
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
    warnings.warn("Scipy not found. Optimization will default to equal weights.")

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'data_dir': "NIFTY100_Sectors_Initial",
    'risk_free_rate': 0.065,  # ~6.5% for India 10Y
    'log_file': "asset_os.log",
    'trading_days': 252
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename=CONFIG['log_file'],
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==========================================
# Helpers
# ==========================================

def clean_percentage_input(user_input):
    """
    Parses messy user input like '20%', '0.20', or '20' into a float.
    Returns 0 if the input implies 'none'.
    """
    if not user_input: return 0.0
    
    clean_str = str(user_input).strip().lower().replace('%', '')
    if clean_str in ["none", "nil", "no", "na"]: 
        return 0.0
    
    try:
        return float(clean_str)
    except ValueError:
        return None

def format_inr(amount):
    """
    Pretty print for Indian Rupees (Lakhs/Crores).
    """
    try:
        val = float(amount)
    except Exception:
        return str(amount)
        
    if val >= 1_000_000_000: # 100 Cr+ (Just in case)
        return f"₹{val/10_000_000:.2f} Cr"
    elif val >= 10_000_000:
        return f"₹{val/10_000_000:.2f} Cr"
    elif val >= 100_000:
        return f"₹{val/100_000:.2f} L"
    else:
        return f"₹{val:,.0f}"

# ==========================================
# Core Engine
# ==========================================

class PortfolioBuilder:
    def __init__(self):
        self.user_profile = {}
        self._market_data_cache = None
        
        # Define the interview flow here so it's easy to edit later
        self.interview_config = [
            {"id": "name", "prompt": "Client Name:", "type": "str"},
            
            {"id": "age", "prompt": "Client Age:", "type": "int"},
            {"id": "income", "prompt": "Annual Income (₹):", "type": "float"},
            {"id": "net_worth", "prompt": "Net Worth (₹):", "type": "float"},
            
            {"id": "goal", "prompt": "Primary Goal (Retirement/Wealth/Income):", "type": "str"},
            {"id": "goal_years", "prompt": "Years to Goal:", "type": "int"},
            
            {"id": "liquidity", "prompt": "Liquidity Needs %:", "type": "pct"},
            
            {"id": "risk_attitude", "prompt": "Risk Attitude (1=Low, 5=High):", "type": "int"},
            {"id": "loss_tolerance", "prompt": "Max Acceptable Loss %:", "type": "pct"},
            
            {"id": "capital", "prompt": "Investment Capital:", "type": "float"},
        ]

    # --- Client Interaction ---
    
    def _get_input(self, prompt_config):
        """Internal helper to handle input validation loops."""
        while True:
            raw = input(f"{prompt_config['prompt']} ").strip()
            
            if not raw and prompt_config['type'] not in ['pct', 'int', 'float']:
                continue
                
            if prompt_config['type'] == 'pct':
                val = clean_percentage_input(raw)
                if val is not None:
                    return val / 100.0 if val > 1 else val
            elif prompt_config['type'] == 'int':
                if raw.isdigit():
                    return int(raw)
            elif prompt_config['type'] == 'float':
                try:
                    return float(raw.replace(',', ''))
                except ValueError:
                    pass
            elif prompt_config['type'] == 'str':
                return raw
                
            print(f"   (!) Please enter a valid {prompt_config['type']}.")

    def run_onboarding(self):
        print("\n" + "-"*50)
        print("   ASSET OS: CLIENT ONBOARDING")
        print("-" * 50)
        
        for item in self.interview_config:
            self.user_profile[item["id"]] = self._get_input(item)
            
        logging.info(f"Onboarding complete for: {self.user_profile.get('name')}")
        return self.user_profile

    # --- Data Layer ---

    def generate_ips(self):
        """
        Generates dynamic investment policy constraints.
        
        ALL onboarding inputs now strongly influence the output constraints.
        Horizon, goal type, age, risk attitude, liquidity, and loss tolerance
        each contribute meaningfully to the final IPS parameters.
        """
        attitude = float(self.user_profile.get('risk_attitude', 3))
        horizon = float(self.user_profile.get('goal_years', 10))
        
        liquidity_raw = float(self.user_profile.get('liquidity', 5))
        liquidity = liquidity_raw / 100.0 if liquidity_raw > 1 else liquidity_raw
        
        loss_raw = float(self.user_profile.get('loss_tolerance', 15))
        loss_tol = loss_raw / 100.0 if loss_raw > 1 else loss_raw
        
        age = float(self.user_profile.get('age', 40))
        goal = str(self.user_profile.get('goal', 'Balanced Income')).strip()
        
        # 1. Composite Risk Score — horizon now has MAJOR weight
        # Horizon factor: short horizons (1-3yr) strongly suppress risk
        if horizon <= 2:
            horizon_factor = 0.10
        elif horizon <= 5:
            horizon_factor = 0.30 + (horizon - 2) * 0.10  # 0.30 to 0.60
        elif horizon <= 10:
            horizon_factor = 0.60 + (horizon - 5) * 0.05  # 0.60 to 0.85
        else:
            horizon_factor = min(1.0, 0.85 + (horizon - 10) * 0.015)  # 0.85+
        
        # Goal-based modifier
        goal_modifier = 0.0
        if 'aggressive' in goal.lower() or 'growth' in goal.lower():
            goal_modifier = 0.15
        elif 'preservation' in goal.lower() or 'conservative' in goal.lower():
            goal_modifier = -0.15
        elif 'income' in goal.lower() or 'balanced' in goal.lower():
            goal_modifier = 0.0
        
        risk_score = (
            (attitude / 5.0) * 0.35 +      # Risk attitude: 35% weight
            horizon_factor * 0.35 +          # Horizon: 35% weight (was 30%)
            max(0, 1 - liquidity) * 0.15 +   # Liquidity: 15% weight
            goal_modifier * 0.15 +            # Goal type: 15% weight (NEW)
            max(0, 1 - age / 80.0) * 0.00    # Age is handled separately below
        )
        risk_score = np.clip(risk_score, 0.05, 0.95)
        
        # 2. Return Objective — wider range, goal-sensitive
        base_return = 0.065  # Risk-free rate baseline
        return_obj = base_return + (risk_score * 0.12)  # Range: 6.5% to 18.5%
        
        # Goal-based return adjustment
        if 'aggressive' in goal.lower() or 'growth' in goal.lower():
            return_obj += 0.03
        elif 'preservation' in goal.lower():
            return_obj -= 0.02
        
        # 3. Equity Allocation Range — much more granular
        if risk_score > 0.75:
            eq_min, eq_max = 0.80, 1.00
        elif risk_score > 0.60:
            eq_min, eq_max = 0.65, 0.85
        elif risk_score > 0.45:
            eq_min, eq_max = 0.50, 0.70
        elif risk_score > 0.30:
            eq_min, eq_max = 0.35, 0.55
        elif risk_score > 0.15:
            eq_min, eq_max = 0.20, 0.40
        else:
            eq_min, eq_max = 0.10, 0.30
        
        # Horizon-based hard override for very short horizons
        if horizon <= 2:
            eq_max = min(eq_max, 0.40)  # Cap equity at 40% for ≤2yr
            eq_min = min(eq_min, 0.15)
        elif horizon <= 3:
            eq_max = min(eq_max, 0.55)  # Cap equity at 55% for ≤3yr
        
        # 4. Age-Based Adjustment — stronger effect
        if age < 30:
            eq_max = min(1.0, eq_max + 0.10)
            eq_min = min(eq_max, eq_min + 0.05)
        elif age < 35:
            eq_max = min(1.0, eq_max + 0.05)
        elif age > 60:
            eq_max = max(eq_min, eq_max - 0.15)
            eq_min = max(0.05, eq_min - 0.10)
        elif age > 50:
            eq_max = max(eq_min, eq_max - 0.05)
        
        # 5. Cash limits — based on liquidity AND horizon
        cash_req = liquidity
        if horizon <= 3:
            cash_req = max(cash_req, 0.10)  # Short horizon → force higher cash
        
        # 6. Max Drawdown — directly from loss tolerance input
        # (this was the correct behavior, but wasn't reaching simulation)
        
        # 7. Output Summary
        print("\n" + "-"*50)
        print("INSTITUTIONAL INVESTMENT POLICY STATEMENT")
        print("-" * 50)
        print(f"Risk Score:        {risk_score:.2f} / 1.0")
        print(f"Target Return:     {return_obj*100:.1f}%")
        print(f"Equity Bounds:     {eq_min*100:.1f}% - {eq_max*100:.1f}%")
        print(f"Cash Requirement:  {cash_req*100:.1f}%")
        print(f"Max Loss Tolerance:{loss_tol*100:.1f}%")
        print(f"Horizon:           {horizon:.0f} years")
        print(f"Goal:              {goal}")
        print("-" * 50)
        
        return {
            "ReturnObjective": return_obj,
            "EquityMin": eq_min,
            "EquityMax": eq_max,
            "CashMin": cash_req,
            "CashMax": min(1.0, cash_req + 0.10),
            "MaxDrawdown": loss_tol
        }

    def load_market_data(self):
        """
        Reads all CSVs in the data folder, aligns dates, and calculates 
        annualized returns (mu) and covariance (cov).
        """
        # Return cached data if we already did the heavy lifting
        if self._market_data_cache:
            return self._market_data_cache

        search_path = os.path.join(CONFIG['data_dir'], "*.csv")
        files = glob.glob(search_path)
        
        if not files:
            logging.error(f"Data directory empty: {CONFIG['data_dir']}")
            raise FileNotFoundError(f"No CSV files found in {CONFIG['data_dir']}. Please check path.")

        # Non-price metadata files that live in the same folder — skip them
        NON_PRICE_FILES = {"investor_profile", "constraints"}

        # Load individual price series
        temp_data = {}
        for f in files:
            if os.path.basename(f).replace('.csv', '') in NON_PRICE_FILES:
                continue
            ticker = os.path.basename(f).replace('.csv', '')
            try:
                # We assume the CSV has dates as index and an 'Adj Close' or 'Close' column
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                
                if target_col in df.columns:
                    # Convert to numeric, coercing any string header rows to NaN
                    temp_data[ticker] = pd.to_numeric(df[target_col], errors='coerce')
                else:
                    logging.warning(f"Column missing in {ticker}, skipping.")
            except Exception as e:
                logging.warning(f"Could not read {ticker}: {e}")

        # Merge, align, and clean
        prices_df = pd.DataFrame(temp_data)
        prices_df = prices_df.ffill().dropna()
        
        if prices_df.empty:
            raise ValueError("Merged price data is empty. Check CSV date alignment.")

        # Calculate Log Returns (better for optimization math)
        # We drop the first row because it becomes NaN after shifting
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        
        # Annualize parameters
        days = CONFIG['trading_days']
        mu = returns_df.mean() * days
        cov = returns_df.cov() * days
        
        self._market_data_cache = (mu, cov, prices_df.columns.tolist())
        return self._market_data_cache

    # --- Math & Optimization ---

    def _optimize(self, expected_returns, cov_matrix, constraints_config=None):
        """
        The core MPT solver. 
        Adjusts limits using IPS engine constraint configurations.
        """
        # Remove Synthetic Assets - Operate only on pure equities
        mu_extended = expected_returns.copy()
        cov_extended = cov_matrix.values
        n_extended = len(mu_extended)
        asset_names = mu_extended.index.tolist()

        if not constraints_config:
            constraints_config = self.generate_ips()

        logging.info(f"Optimizer constraints: {constraints_config}")

        bounds = tuple((0.0, 1.0) for _ in range(n_extended))

        # Constraints (Equity sums leaving explicit room for Cash)
        constraints = [
            {"type": "ineq", "fun": lambda w: np.sum(w) - constraints_config['EquityMin']},
            {"type": "ineq", "fun": lambda w: constraints_config['EquityMax'] - np.sum(w)},
            {"type": "ineq", "fun": lambda w: (1.0 - np.sum(w)) - constraints_config['CashMin']},
            {"type": "ineq", "fun": lambda w: constraints_config['CashMax'] - (1.0 - np.sum(w))}
        ]

        def objective_function(w):
            port_ret = np.dot(w, mu_extended)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_extended, w)))
            
            if port_vol == 0: return 0 
            
            sharpe = (port_ret - CONFIG['risk_free_rate']) / port_vol
            
            target_return = constraints_config['ReturnObjective']
            risk_threshold = constraints_config['MaxDrawdown']
            
            penalty_return = max(0, target_return - port_ret)**2
            penalty_risk = max(0, port_vol - risk_threshold)**2
            
            objective = -sharpe + 10*penalty_return + 5*penalty_risk
            return objective

        # Base distributed guess matching minimum equity
        init_guess = np.ones(n_extended) * (constraints_config['EquityMin'] / n_extended)
        
        if minimize:
            res = minimize(
                objective_function, 
                init_guess, 
                method="SLSQP", 
                bounds=bounds, 
                constraints=constraints
            )
            optimal_weights = res.x if res.success else init_guess
            if not res.success:
                logging.warning("Optimization failed to converge. Returning initial guess.")
        else:
            optimal_weights = init_guess

        # Set dynamic cash value based on residual portfolio weight
        actual_cash = 1.0 - np.sum(optimal_weights)
        
        final_alloc = pd.Series(optimal_weights, index=asset_names)
        final_alloc["CASH_RESERVE"] = max(0.0, actual_cash)
        
        return final_alloc.sort_values(ascending=False)

    def run_black_litterman(self, views_map):
        """
        Adjusts the market equilibrium (historical data) based on user views.
        This is a simplified implementation of the Black-Litterman Formula.
        """
        print("\n[Analysis] Integrating user views into Black-Litterman model...")
        
        # Load market data
        _, cov, assets = self.load_market_data()
        
        # Risk Aversion and Tau parameters
        delta = 2.5
        tau = 0.05
        
        n_assets = len(assets)
        n_views = len(views_map)
        
        if n_views == 0:
            print("   (!) No views provided. Reverting to standard MPT.")
            return self.run_mpt_only()
            
        # 1. Market Weights (Equal weights proxy)
        w = np.ones(n_assets) / n_assets
        
        # 2. Implied Equilibrium Returns (Pi)
        pi = delta * np.dot(cov.values, w)
        
        # 3. Build the Pick Matrix (P) and View Vector (Q)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        view_keys = list(views_map.keys())
        
        for i, ticker in enumerate(view_keys):
            if ticker in assets:
                col_idx = assets.index(ticker)
                P[i, col_idx] = 1        # Absolute view
                Q[i] = views_map[ticker] # Expected return
        
        # 4. Compute View Uncertainty Matrix (Omega)
        # Omega = diag(P * (tau * Sigma) * P.T)
        view_variances = np.diag(np.dot(np.dot(P, tau * cov.values), P.T))
        omega = np.diag(view_variances)
        
        # 5. Compute Black-Litterman Expected Returns (mu_bl)
        try:
            # We use pseudo-inverse (pinv) to be safe against singular matrices
            ts_inv = np.linalg.pinv(tau * cov.values)
            omega_inv = np.linalg.pinv(omega)
            
            term_1 = np.linalg.inv(ts_inv + np.dot(np.dot(P.T, omega_inv), P))
            term_2 = np.dot(ts_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
            
            mu_bl = np.dot(term_1, term_2)
            
        except np.linalg.LinAlgError as e:
            logging.error(f"Matrix inversion failed in BL model: {e}")
            print("   (!) Error in math calculation. Reverting to standard MPT.")
            return self.run_mpt_only()

        # Re-run the optimizer with the NEW expected returns, but OLD covariance
        # (We assume views change returns, not risk)
        bl_mu_series = pd.Series(mu_bl, index=assets)
        
        constraints = self.generate_ips()
        return self._optimize(bl_mu_series, cov, constraints)

    # def run_monte_carlo(self, allocation):
    #     """
    #     Runs a simulation to determine VaR (Value at Risk).
    #     """
    #     print("\n[Stress Test] Spinning up 1,000 simulations...")
    #     mu, cov, assets = self.load_market_data()
        
    #     # Strip out cash for the simulation, we only simulate the risky assets
    #     risky_alloc = allocation.drop("CASH_RESERVE", errors='ignore')
        
    #     total_equity_weight = risky_alloc.sum()
    #     if total_equity_weight < 0.01:
    #         print("   (!) Portfolio is effectively 100% Cash. Skipping Monte Carlo.")
    #         return

    #     # Normalize weights to 100% of the *equity* portion for the math to work
    #     weights_norm = risky_alloc / total_equity_weight
        
    #     # Get portfolio level stats
    #     # Ensure we align indices
    #     aligned_mu = mu[weights_norm.index]
    #     aligned_cov = cov.loc[weights_norm.index, weights_norm.index]
        
    #     port_ret = np.dot(weights_norm, aligned_mu)
    #     port_vol = np.sqrt(np.dot(weights_norm.T, np.dot(aligned_cov, weights_norm)))
        
    #     # Simulation Parameters
    #     sim_count = 1000
    #     days = 252
        
    #     # Generate random Z-scores
    #     # Daily expected return and daily volatility
    #     daily_mu = port_ret / days
    #     daily_vol = port_vol / np.sqrt(days)
        
    #     sim_matrix = pd.DataFrame()
        
    #     # Run loops (vectorized where possible, but loop for path dependency)
    #     for i in range(sim_count):
    #         random_shocks = np.random.normal(0, 1, days)
    #         price_path = [100] # Start at index 100
            
    #         for shock in random_shocks:
    #             # Geometric Brownian Motion formula
    #             drift = daily_mu - 0.5 * (daily_vol ** 2)
    #             diffusion = daily_vol * shock
    #             price_path.append(price_path[-1] * np.exp(drift + diffusion))
                
    #         sim_matrix[f"sim_{i}"] = price_path

    #     # Analyze Results
    #     end_values = sim_matrix.iloc[-1]
        
    #     var_95 = 100 - np.percentile(end_values, 5) # 5th percentile
    #     median_case = np.percentile(end_values, 50)
    #     upside_case = np.percentile(end_values, 95)
        
    #     print(f"   -> 95% Value at Risk (VaR): -{var_95:.2f}%")
    #     print(f"   -> Median Outcome: +{median_case - 100:.2f}%")
    #     print(f"   -> Optimistic Outcome: +{upside_case - 100:.2f}%")
        
    #     # Quick Plot
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(sim_matrix, color='blue', alpha=0.05, linewidth=1)
    #     plt.plot(sim_matrix.mean(axis=1), color='red', linewidth=2, label='Mean Path')
    #     plt.title(f"Monte Carlo Simulation (Equity Only) - {sim_count} Iterations")
    #     plt.xlabel("Trading Days")
    #     plt.ylabel("Portfolio Index")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    def run_monte_carlo(self, allocation):
        """
        Runs 1,000 simulations but visualizes them as a 'Cone of Uncertainty'
        so normal people can understand the risk/reward.
        """
        print("\n[Risk Analysis] Running stress test...")
        mu, cov, assets = self.load_market_data()
        
        # 1. Prepare Weights — drop all synthetic / non-market assets
        SYNTHETIC = ["CASH_RESERVE", "BOND_INDEX", "ALT_INDEX"]
        risky_alloc = allocation.drop(SYNTHETIC, errors='ignore')
        total_equity = risky_alloc.sum()
        
        if total_equity < 0.01:
            print("   (!) Portfolio is 100% Cash. No risk to simulate.")
            return

        weights_norm = risky_alloc / total_equity
        
        # 2. Portfolio Stats
        aligned_mu = mu[weights_norm.index]
        aligned_cov = cov.loc[weights_norm.index, weights_norm.index]
        
        port_ret = np.dot(weights_norm, aligned_mu)
        port_vol = np.sqrt(np.dot(weights_norm.T, np.dot(aligned_cov, weights_norm)))
        
        # 3. Simulation — fully vectorized (no Python loop)
        sim_count = 400
        days = 252 * 5  # Simulate 5 Years

        daily_mu = port_ret / 252
        daily_vol = port_vol / np.sqrt(252)
        drift = daily_mu - 0.5 * (daily_vol ** 2)

        shocks = np.random.normal(0, 1, (days - 1, sim_count))
        daily_returns = np.exp(drift + daily_vol * shocks)
        cum_returns = np.vstack([
            np.ones((1, sim_count)),
            np.cumprod(daily_returns, axis=0)
        ])
        sim_matrix = 100 * cum_returns

        # 4. Process Data for Humans (The "Cone")
        # Instead of plotting lines, we calculate percentiles for every day
        median_path = np.percentile(sim_matrix, 50, axis=1)
        top_10 = np.percentile(sim_matrix, 90, axis=1)
        bottom_10 = np.percentile(sim_matrix, 10, axis=1)
        
        # Get Investment Amount for context (Default to 1 Lakh if missing)
        try:
            capital = float(self.user_profile.get('capital', 100000))
        except Exception:
            capital = 100000
            
        final_mult = sim_matrix[-1, :] / 100
        worst_case = np.percentile(final_mult, 5) * capital
        best_case = np.percentile(final_mult, 95) * capital
        
        # 5. Output in Plain English
        print(f"\n FORECAST FOR NEXT 5 YEARS (Based on ₹{capital:,.0f}) ")
        print(f" Worst Case (Bear Market):  ₹{worst_case:,.0f}")
        print(f" Best Case (Bull Market):  ₹{best_case:,.0f}")
        print(f" Likely Outcome:  ₹{np.median(final_mult) * capital:,.0f}")
        
        # 6. The "Clean" Plot
        plt.figure(figsize=(10, 6))
        
        # Plot the "Cone"
        x = np.arange(days)
        plt.fill_between(x, bottom_10, top_10, color='blue', alpha=0.2, label='Likely Range (80% confidence)')
        plt.plot(x, median_path, color='navy', linewidth=2, label='Average Growth Path')
        
        # Add a few "example" lines just to show it's a simulation, but faint
        plt.plot(x, sim_matrix[:, :3], color='gray', alpha=0.3, linewidth=1)
        
        plt.title(f"Projected Growth: {total_equity*100:.0f}% Equity / {100-total_equity*100:.0f}% Cash")
        plt.xlabel("Trading Days (5 Years)")
        plt.ylabel("Portfolio Index (Start = 100)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_api_onboarding(self, user_data):
        """
        API version of onboarding. Accepts a dictionary directly.
        """
        # Map frontend keys to internal keys if necessary, or just use raw
        # Expected keys: name, goal, liquidity, capital, risk_tolerance
        self.user_profile = user_data
        logging.info(f"API Onboarding complete for: {self.user_profile.get('name')}")
        return self.user_profile

    def run_full_simulation(self, allocation, horizon_years=None, capital=None):
        """
        Full Monte Carlo simulation engine with dynamic horizon.
        
        EVERY onboarding input influences this output:
        - horizon_years: determines simulation length → affects compounding,
          wealth dispersion, volatility accumulation, and confidence intervals
        - capital: scales all outputs to real ₹ values
        - Portfolio weights: determine expected return and volatility
        
        Returns 5 percentile bands, sample paths, and computed portfolio metrics.
        """
        mu, cov, assets = self.load_market_data()
        
        # --- Resolve horizon from user profile if not passed ---
        if horizon_years is None:
            horizon_years = float(self.user_profile.get('goal_years', 5))
        horizon_years = max(1, min(30, horizon_years))  # Clamp to 1-30 years
        
        if capital is None:
            try:
                capital = float(str(self.user_profile.get('capital', 100000)).replace(',', '').replace('$', ''))
            except Exception:
                capital = 100000
        
        # 1. Prepare Weights — drop all synthetic / non-market assets
        SYNTHETIC = ["CASH_RESERVE", "BOND_INDEX", "ALT_INDEX"]
        risky_alloc = allocation.drop(SYNTHETIC, errors='ignore')
        total_equity = risky_alloc.sum()
        cash_weight = allocation.get("CASH_RESERVE", 0.0)
        
        if total_equity < 0.01:
            return {"error": "Portfolio is 100% Cash. No risk to simulate."}

        weights_norm = risky_alloc / total_equity
        
        # 2. Portfolio Stats — these are REAL computed values
        aligned_mu = mu[weights_norm.index]
        aligned_cov = cov.loc[weights_norm.index, weights_norm.index]
        
        port_ret = float(np.dot(weights_norm, aligned_mu))
        port_vol = float(np.sqrt(np.dot(weights_norm.T, np.dot(aligned_cov, weights_norm))))
        
        # --- Risk-free rate ---
        rf = CONFIG['risk_free_rate']
        
        # 3. Simulation — HORIZON-DEPENDENT
        sim_count = 1000
        days = int(252 * horizon_years)
        
        daily_mu = port_ret / 252
        daily_vol = port_vol / np.sqrt(252)
        drift = daily_mu - 0.5 * (daily_vol ** 2)
        
        # ---- Regime-switching volatility for realism ----
        # 3 regimes: calm (0.7x vol), normal (1.0x), stressed (1.8x vol)
        # Markov transition: each day has small chance of regime change
        regime_vols = np.ones(days - 1)
        current_regime = 1  # 0=calm, 1=normal, 2=stressed
        regime_multipliers = [0.7, 1.0, 1.8]
        transition_probs = [
            [0.97, 0.025, 0.005],  # From calm
            [0.015, 0.97, 0.015],  # From normal
            [0.01, 0.04, 0.95],    # From stressed
        ]
        
        regime_rng = np.random.random(days - 1)
        for d in range(days - 1):
            probs = transition_probs[current_regime]
            if regime_rng[d] < probs[0]:
                current_regime = 0
            elif regime_rng[d] < probs[0] + probs[1]:
                current_regime = 1
            else:
                current_regime = 2
            regime_vols[d] = regime_multipliers[current_regime]
        
        # Generate shocks with regime-adjusted volatility
        # Each simulation gets the SAME regime sequence (market is shared)
        # but different random shocks (individual paths differ)
        base_shocks = np.random.normal(0, 1, (days - 1, sim_count))
        adjusted_vols = (daily_vol * regime_vols).reshape(-1, 1)  # (days-1, 1)
        daily_returns = np.exp(drift + adjusted_vols * base_shocks)
        
        # Compute cumulative paths — START FROM CAPITAL, not index 100
        cum_returns = np.vstack([
            np.ones((1, sim_count)),
            np.cumprod(daily_returns, axis=0)
        ])
        sim_matrix = capital * cum_returns  # Shape: (days, sim_count)
        # Cash portion grows at risk-free rate
        cash_growth = capital * cash_weight * ((1 + rf) ** (np.arange(days) / 252)).reshape(-1, 1)
        equity_portion = total_equity
        sim_matrix = sim_matrix * equity_portion + cash_growth
        
        # 4. Calculate 5 Percentile Bands
        p5 = np.percentile(sim_matrix, 5, axis=1)
        p25 = np.percentile(sim_matrix, 25, axis=1)
        p50 = np.percentile(sim_matrix, 50, axis=1)
        p75 = np.percentile(sim_matrix, 75, axis=1)
        p95 = np.percentile(sim_matrix, 95, axis=1)
        
        # 5. Sample individual paths for visual authenticity (6 paths)
        sample_indices = np.random.choice(sim_count, size=min(6, sim_count), replace=False)
        sample_paths = sim_matrix[:, sample_indices]
        
        # 6. Compute portfolio metrics from simulation
        # Max Drawdown — compute for each path, take median
        running_max = np.maximum.accumulate(sim_matrix, axis=0)
        drawdowns = (running_max - sim_matrix) / running_max
        max_drawdowns_per_path = np.max(drawdowns, axis=0)
        median_max_drawdown = float(np.median(max_drawdowns_per_path))
        p95_max_drawdown = float(np.percentile(max_drawdowns_per_path, 95))
        
        # Sharpe Ratio (annualized)
        sharpe = (port_ret - rf) / port_vol if port_vol > 1e-8 else 0.0
        
        # Sortino Ratio — downside deviation only (per-path, then average)
        daily_rets_all = np.diff(sim_matrix, axis=0) / sim_matrix[:-1]
        downside_rets = daily_rets_all.copy()
        downside_rets[downside_rets > 0] = 0
        # Compute per-path downside vol, then take mean across paths
        per_path_downside_vol = np.std(downside_rets, axis=0) * np.sqrt(252)
        downside_vol = float(np.mean(per_path_downside_vol))
        sortino = (port_ret - rf) / downside_vol if downside_vol > 1e-8 else 0.0
        
        # Diversification Ratio
        individual_vols = np.sqrt(np.diag(aligned_cov.values))
        weighted_vol_sum = float(np.dot(weights_norm, individual_vols))
        diversification_ratio = weighted_vol_sum / port_vol if port_vol > 1e-8 else 1.0
        
        # VaR and CVaR — annualized from DAILY returns (not terminal)
        # This prevents VaR from showing -133% on profitable long-horizon portfolios
        terminal_returns = (sim_matrix[-1] / capital) - 1  # Keep for final_stats
        daily_rets_flat = daily_rets_all.flatten()  # All daily returns across all paths
        annualized_daily_var = float(-np.percentile(daily_rets_flat, 5) * np.sqrt(252))
        tail_daily = daily_rets_flat[daily_rets_flat <= np.percentile(daily_rets_flat, 5)]
        annualized_daily_cvar = float(-np.mean(tail_daily) * np.sqrt(252)) if len(tail_daily) > 0 else annualized_daily_var
        # Clamp to reasonable range (0% to 100%)
        var_95 = max(0.0, min(annualized_daily_var, 1.0))
        cvar_95 = max(0.0, min(annualized_daily_cvar, 1.0))
        
        # Liquidity score — based on cash allocation
        liquidity_score = min(1.0, cash_weight * 5)  # 20% cash = perfect liquidity
        
        # 7. Downsample for JSON response — monthly snapshots
        trading_days_per_month = 21
        step = max(1, trading_days_per_month)
        indices = list(range(0, days, step))
        if (days - 1) not in indices:
            indices.append(days - 1)  # Always include final day
        
        # Convert day indices to clean month numbers
        months = [int(round(d / 21)) for d in indices]
        
        # --- Scorecard Calculation ---
        risk_score_raw = min(100, max(0, sharpe * 50))
        downside_score_raw = max(0, 100 - (median_max_drawdown * 100 * 2.5))
        div_score_raw = min(100, (diversification_ratio - 1) * 100) if diversification_ratio > 1 else 0
        liq_score_raw = liquidity_score * 100
        
        # Get target return from constraints
        constraints_config = self.generate_ips()
        target_return = constraints_config['ReturnObjective']
        goal_score_raw = 100 if port_ret >= target_return else max(0, 100 - ((target_return - port_ret) * 100 * 10))
        
        overall_score_raw = (risk_score_raw * 0.3) + (downside_score_raw * 0.25) + (div_score_raw * 0.15) + (liq_score_raw * 0.10) + (goal_score_raw * 0.20)
        
        def get_grade(score):
            if score >= 90: return "A+"
            if score >= 80: return "A"
            if score >= 75: return "A-"
            if score >= 70: return "B+"
            if score >= 60: return "B"
            if score >= 50: return "C"
            return "D"
            
        scorecard = {
            "risk_adjusted": {"score": round(risk_score_raw), "grade": get_grade(risk_score_raw)},
            "downside": {"score": round(downside_score_raw), "grade": get_grade(downside_score_raw)},
            "diversification": {"score": round(div_score_raw), "grade": get_grade(div_score_raw)},
            "liquidity": {"score": round(liq_score_raw), "grade": get_grade(liq_score_raw)},
            "goal_alignment": {"score": round(goal_score_raw), "grade": get_grade(goal_score_raw)},
            "overall": {"score": round(overall_score_raw, 1), "grade": get_grade(overall_score_raw)}
        }
        
        # --- Benchmarks Calculation ---
        # Nifty proxy: 12% return, 18% vol
        nifty_mu = 0.12
        nifty_vol = 0.18
        nifty_drift = (nifty_mu - 0.5 * nifty_vol**2) * horizon_years
        nifty_wealth = capital * np.exp(nifty_drift)
        
        fd_wealth = capital * (1 + 0.065) ** horizon_years
        savings_wealth = capital * (1 + 0.035) ** horizon_years
        
        benchmarks = {
            "nifty_50": {"wealth": round(nifty_wealth), "return_pct": round((nifty_wealth/capital - 1)*100, 1), "max_dd": 40.0, "sharpe": 0.55},
            "fixed_deposit": {"wealth": round(fd_wealth), "return_pct": round((fd_wealth/capital - 1)*100, 1), "max_dd": 0.0, "sharpe": "N/A"},
            "savings": {"wealth": round(savings_wealth), "return_pct": round((savings_wealth/capital - 1)*100, 1), "max_dd": 0.0, "sharpe": "N/A"}
        }
        
        # --- Goal Probability ---
        target_wealth = capital * (1 + target_return) ** horizon_years
        goal_probability = float(np.mean(sim_matrix[-1] >= target_wealth) * 100)
        
        return {
            "months": months,
            "horizon_years": horizon_years,
            "capital": capital,
            "p5": [round(float(p5[i]), 2) for i in indices],
            "p25": [round(float(p25[i]), 2) for i in indices],
            "median": [round(float(p50[i]), 2) for i in indices],
            "p75": [round(float(p75[i]), 2) for i in indices],
            "p95": [round(float(p95[i]), 2) for i in indices],
            "sample_paths": [
                [round(float(sample_paths[i, j]), 2) for i in indices]
                for j in range(sample_paths.shape[1])
            ],
            "final_stats": {
                "worst_case": round(float(np.percentile(sim_matrix[-1], 5)), 2),
                "best_case": round(float(np.percentile(sim_matrix[-1], 95)), 2),
                "median_case": round(float(np.median(sim_matrix[-1])), 2),
                "worst_case_pct": round(float(np.percentile(terminal_returns, 5) * 100), 2),
                "best_case_pct": round(float(np.percentile(terminal_returns, 95) * 100), 2),
                "median_pct": round(float(np.median(terminal_returns) * 100), 2),
            },
            "portfolio_metrics": {
                "expected_return": round(port_ret * 100, 2),
                "volatility": round(port_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "max_drawdown": round(median_max_drawdown * 100, 2),
                "max_drawdown_p95": round(p95_max_drawdown * 100, 2),
                "diversification_ratio": round(diversification_ratio, 3),
                "liquidity_score": round(liquidity_score * 100, 1),
                "var_95": round(var_95 * 100, 2),
                "cvar_95": round(cvar_95 * 100, 2),
                "equity_weight": round(total_equity * 100, 2),
                "cash_weight": round(cash_weight * 100, 2),
                "n_assets": int((risky_alloc > 0.001).sum()),
            },
            "simulation_info": {
                "n_paths": sim_count,
                "horizon_days": days,
                "horizon_years": horizon_years,
            },
            "scorecard": scorecard,
            "benchmarks": benchmarks,
            "goal_probability": round(goal_probability, 1)
        }

    # Legacy alias for backward compatibility
    def get_monte_carlo_data(self, allocation):
        """Legacy wrapper — calls run_full_simulation with profile defaults."""
        return self.run_full_simulation(allocation)

    def run_mpt_only(self):
        """Wrapper to just run standard MPT without views."""
        mu, cov, _ = self.load_market_data()
        
        constraints = self.generate_ips()
        return self._optimize(mu, cov, constraints)


# ==========================================
# Main Execution
# ==========================================

def main():
    app = PortfolioBuilder()
    
    # 1. Onboarding
    profile = app.run_onboarding()
    client_name = profile.get('name', 'Client').upper()
    
    # 2. Baseline Generation
    print("\nGenerating Baseline Strategy ")
    try:
        base_weights = app.run_mpt_only()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not run MPT. {e}")
        return

    print(f"\nBaseline for {client_name}:")
    print("-" * 30)
    # Filter out tiny weights for cleaner printing
    print((base_weights[base_weights > 0.001] * 100).round(2).astype(str) + " %")
    
    # 3. Intelligent Adjustment (Human + AI Loop)
    print("\n" + "="*50)
    print("   PHASE 2: INTELLIGENT ADJUSTMENT")
    print("="*50)
    
    _, _, valid_tickers = app.load_market_data()
    print(f"Available Assets: {', '.join(valid_tickers[:10])} ...") # Just show first 10
    
    do_adjust = input("\nWould you like to inject market views? (y/n): ").strip().lower()
    
    final_weights = base_weights
    method_used = "Standard MPT"

    if do_adjust.startswith('y'):
        user_views = {}
        print("\nInput format: Ticker, then Sentiment. Type 'DONE' to finish.")
        
        while True:
            t_input = input("   Asset Ticker: ").strip().upper()
            if t_input == 'DONE': break
            
            # Fuzzy match / Validation
            found_ticker = next((x for x in valid_tickers if x in t_input), None)
            
            if not found_ticker:
                print(f"   (?) Could not find '{t_input}' in data. Try again.")
                continue
            
            print(f"   > Sentiment for {found_ticker}?")
            print("     1. Bullish (Expect +25%)")
            print("     2. Neutral (Expect +12%)")
            print("     3. Bearish (Expect -5%)")
            
            try:
                choice = int(input("     Selection (1-3): "))
                if choice == 1: val = 0.25
                elif choice == 3: val = -0.05
                else: val = 0.12
                
                user_views[found_ticker] = val
                print(f"     Recorded view on {found_ticker}")
            except ValueError:
                print("     Invalid choice. Skipping.")

        if user_views:
            final_weights = app.run_black_litterman(user_views)
            method_used = "Black-Litterman (AI Adjusted)"
            
            print("\n" + "-"*40)
            print("   FINAL OPTIMIZED PORTFOLIO")
            print("-" * 40)
            print((final_weights[final_weights > 0.001] * 100).round(2).astype(str) + " %")

    # 4. Stress Test
    print("\n[Phase 3] Risk Analysis")
    sim_data = app.run_full_simulation(final_weights)
    fs = sim_data.get('final_stats', {})
    pm = sim_data.get('portfolio_metrics', {})
    print(f"\n   Worst Case:  ₹{fs.get('worst_case', 0):,.0f}")
    print(f"   Likely:     ₹{fs.get('median_case', 0):,.0f}")
    print(f"   Best Case:  ₹{fs.get('best_case', 0):,.0f}")
    print(f"   Sharpe:     {pm.get('sharpe_ratio', 0):.3f}")
    print(f"   Max DD:     {pm.get('max_drawdown', 0):.1f}%")

    # 5. Output
    filename = f"Portfolio_{client_name.replace(' ', '_')}.csv"
    final_weights.to_csv(filename, header=["Weight"])
    
    print("\n" + "="*50)
    print("   SUCCESS")
    print(f"   Client:   {client_name}")
    print(f"   Strategy: {method_used}")
    print(f"   Output:   {filename}")
    print("="*50)


if __name__ == "__main__":
    main()
