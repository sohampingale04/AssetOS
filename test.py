# asset_os_test_suite.py
# Run this to generate 3 distinct portfolios based on different user personas

def run_test_suite(app):
    test_cases = [
        {
            "name": "TEST_CASE_1_AGGRESSIVE",
            "liquidity_needs": 0,
            "desc": "Young, High Risk, 0% Cash"
        },
        {
            "name": "TEST_CASE_2_CONSERVATIVE",
            "liquidity_needs": 25,
            "desc": "Retiree, Low Risk, 25% Cash Buffer"
        },
        {
            "name": "TEST_CASE_3_BALANCED",
            "liquidity_needs": 10,
            "desc": "Mid-Career, Moderate Risk, 10% Cash"
        }
    ]

    print("\n" + "="*60)
    print("      ASSET OS: AUTOMATED TEST SUITE")
    print("="*60)

    # Load Data Once
    mu, cov, assets = app.get_market_data()
    
    for case in test_cases:
        print(f"\nrunning >> {case['name']} ({case['desc']})")
        
        # 1. Simulate Profile
        fake_profile = {
            "name": case["name"], 
            "liquidity_needs": case["liquidity_needs"]
        }
        
        # 2. Run Optimizer
        # Note: We pass the pre-loaded data to save time
        # We need to slightly mod run_mpt to accept data if we want speed, 
        # but calling it normally is fine for 3 tests.
        
        # Let's just manually call the logic here to show the difference
        n = len(assets)
        # ... (standard optimization logic hidden for brevity) ...
        # For the test suite, we just want to see the Allocation Swing
        
        weights = app.run_mpt(fake_profile)
        
        # 3. Print Top 3 Holdings
        print("-" * 40)
        print(f"CASH RESERVE: {weights['CASH_RESERVE']*100:.1f}%")
        print("TOP 3 HOLDINGS:")
        top_3 = weights.drop("CASH_RESERVE").head(3)
        for ticker, w in top_3.items():
            print(f"  - {ticker}: {w*100:.2f}%")
        print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize your app class from the previous code
    app = DynamicIPS_MPT() 
    run_test_suite(app)