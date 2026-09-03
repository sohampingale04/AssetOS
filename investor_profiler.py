import pandas as pd
import os

class InvestorProfiler:
    def __init__(self, data_path="AssetOS_Data/investor_profile.csv"):
        self.data_path = data_path
        self.profiles = pd.DataFrame()
        self.constraints = {}

    def load_profiles(self):
        """Reads the investor_profile dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Investor profile dataset not found at {self.data_path}")
        self.profiles = pd.read_csv(self.data_path)
        return self.profiles

    def generate_constraints(self):
        """Iterates over profiles and generates allocation bounds based on rules."""
        if self.profiles.empty:
            self.load_profiles()

        for _, row in self.profiles.iterrows():
            investor_id = int(row['InvestorID'])
            risk = str(row['RiskTolerance']).strip().lower()
            horizon = float(row['InvestmentHorizon'])
            liquidity = str(row['LiquidityNeed']).strip().lower()

            # 1. Base Allocation by Risk Tolerance
            if risk == 'conservative':
                bounds = {
                    'EquityMin': 0.20, 'EquityMax': 0.40,
                    'BondMin': 0.40, 'BondMax': 0.60,
                    'AltMin': 0.00, 'AltMax': 0.10,
                    'CashMin': 0.05, 'CashMax': 0.20
                }
            elif risk == 'aggressive':
                bounds = {
                    'EquityMin': 0.60, 'EquityMax': 0.80,
                    'BondMin': 0.05, 'BondMax': 0.25,
                    'AltMin': 0.10, 'AltMax': 0.20,
                    'CashMin': 0.00, 'CashMax': 0.05
                }
            else: # moderate / default
                bounds = {
                    'EquityMin': 0.40, 'EquityMax': 0.60,
                    'BondMin': 0.20, 'BondMax': 0.40,
                    'AltMin': 0.05, 'AltMax': 0.15,
                    'CashMin': 0.00, 'CashMax': 0.10
                }

            # 2. Investment Horizon Adjustments
            if horizon > 20:
                bounds['EquityMax'] += 0.05
                bounds['BondMax'] = max(0, bounds['BondMax'] - 0.05)
            elif horizon < 10:
                bounds['EquityMax'] = max(0, bounds['EquityMax'] - 0.10)
                bounds['BondMin'] += 0.10

            # 3. Liquidity Need Adjustments
            if liquidity == 'high':
                bounds['CashMin'] = 0.10
                bounds['CashMax'] = 0.20
            elif liquidity == 'medium':
                bounds['CashMin'] = 0.05
                bounds['CashMax'] = 0.10
            elif liquidity == 'low':
                bounds['CashMin'] = 0.00
                bounds['CashMax'] = 0.05

            # Sanity checks to ensure Min <= Max
            for asset_class in ['Equity', 'Bond', 'Alt', 'Cash']:
                bounds[f'{asset_class}Min'] = min(bounds[f'{asset_class}Min'], bounds[f'{asset_class}Max'])

            self.constraints[investor_id] = bounds

        return self.constraints

    def export_constraints(self, output_path="AssetOS_Data/constraints.csv"):
        """Exports constraints to a CSV file."""
        if not self.constraints:
            self.generate_constraints()
        
        df = pd.DataFrame.from_dict(self.constraints, orient='index')
        df.index.name = 'InvestorID'
        df.to_csv(output_path)
        print(f"Constraints generated and saved to {output_path}")

    def get_constraints_for_investor(self, investor_id):
        """Returns constraints dictionary for a specific investor."""
        if not self.constraints:
            self.generate_constraints()
        return self.constraints.get(investor_id, None)

if __name__ == "__main__":
    # Test script functionally
    profiler = InvestorProfiler()
    constraints = profiler.generate_constraints()
    profiler.export_constraints()
    
    for i_id, c in constraints.items():
        print(f"Investor {i_id}: {c}")
