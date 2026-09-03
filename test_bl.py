import os
import sys

# Setup environment to import correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPS import PortfolioBuilder

def test_black_litterman():
    app = PortfolioBuilder()
    
    # 1. Provide minimal user profile
    app.user_profile = {
        'name': 'Test User',
        'liquidity': 10  # 10% cash buffer
    }
    
    print("\n--- Testing Baseline MPT ---")
    base_weights = app.run_mpt_only()
    print(base_weights[base_weights > 0.001].round(4))
    
    print("\n--- Testing Black-Litterman ---")
    views = {
        "TCS": 0.30,      # Extremely bullish on TCS
        "RELIANCE": -0.10 # Bearish on Reliance
    }
    
    try:
        bl_weights = app.run_black_litterman(views)
        print(bl_weights[bl_weights > 0.001].round(4))
        
        print("\nSuccess! The model successfully completed optimization and adjusted weights.")
    except Exception as e:
        print(f"Error during Black-Litterman optimization: {e}")

if __name__ == "__main__":
    test_black_litterman()
