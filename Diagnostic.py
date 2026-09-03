# ==========================================
# DIAGNOSTIC TOOL
# ==========================================
def run_diagnostic():
    print("\n--- ASSET OS: DATA AUDIT ---\n")
    
    # 1. Load Data
    files = glob.glob("AssetOS_Data/*.csv")
    if not files: return print("❌ No Data Found")
    
    price_data = {}
    for file in files:
        ticker = os.path.basename(file).split('.')[0]
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            price_data[ticker] = df[col]
        except: pass
        
    prices = pd.DataFrame(price_data).ffill().dropna()
    
    # 2. Calculate Hard Numbers
    # Annualized Returns (CAGR approx)
    returns = np.log(prices / prices.shift(1)).dropna()
    mu = returns.mean() * 252
    
    # Risk Free Rate used in code
    rf = 0.065 
    
    # 3. TEST 1: The "Risk Premium" Check
    print(f"Check 1: Are stocks actually beating the Risk-Free Rate ({rf*100}%)?")
    print("-" * 60)
    print(f"{'TICKER':<15} {'RETURN (Ann)':<15} {'STATUS'}")
    print("-" * 60)
    
    bad_assets = 0
    for ticker, ret in mu.items():
        status = "✅ Healthy" if ret > rf else "⚠️ LOW RETURN"
        if ret <= rf: bad_assets += 1
        print(f"{ticker:<15} {ret*100:.2f}%          {status}")
    
    print("-" * 60)
    if bad_assets > 0:
        print(f"\n🚨 DIAGNOSIS: {bad_assets} of your {len(mu)} assets have returns LOWER than the Bank FD Rate.")
        print("   -> EFFECT: The Optimizer will hate these stocks. It forces the graph to look distorted.")
    else:
        print("\n✅ DIAGNOSIS: All assets are profitable. This is good.")

    # 4. TEST 2: The "Correlation" Check
    print("\n\nCheck 2: Are your sectors just copying each other?")
    corr_matrix = prices.pct_change().corr()
    avg_corr = corr_matrix.mean().mean()
    
    print(f"Average Correlation in Portfolio: {avg_corr:.2f} (0=Diverse, 1.0=Identical)")
    
    if avg_corr > 0.5:
        print("🚨 HIGH CORRELATION DETECTED.")
        print("   -> EFFECT: Your Efficient Frontier will look like a 'Straight Line' instead of a curve.")
        print("   -> REASON: IT and Pharma often move together. Diversification power is weak.")
    else:
        print("✅ Correlation is low. The curve should look nice and wide.")

run_diagnostic()