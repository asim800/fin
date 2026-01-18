"""
CVaR Portfolio Optimization using Rockafellar-Uryasev Formulation

Maximize CVaR (equivalently, minimize left-tail risk) for a portfolio
while extracting VaR as a byproduct.
"""

import numpy as np
import cvxpy as cp

# -----------------------------
# Generate sample return scenarios
# -----------------------------

np.random.seed(42)
n_scenarios = 1000
n_assets = 4

# Asset characteristics (annualized)
# SPX, Bonds, Gold, Tail Hedge (negative expected return but positive in crashes)
expected_returns = np.array([0.10, 0.04, 0.05, -0.15])
volatilities = np.array([0.18, 0.06, 0.15, 0.40])

# Correlation matrix
# Tail hedge has strong negative correlation with SPX during stress
corr_matrix = np.array([
    [1.0,  0.1,  0.0, -0.3],
    [0.1,  1.0,  0.2,  0.0],
    [0.0,  0.2,  1.0,  0.1],
    [-0.3, 0.0,  0.1,  1.0]
])

# Build covariance matrix
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# Generate normal scenarios
normal_returns = np.random.multivariate_normal(
    expected_returns, cov_matrix, size=n_scenarios
)

# Add some crash scenarios to make it interesting
# Spitznagel's point: rare but SEVERE crashes justify tail hedges
# Let's make crashes rarer but more devastating
n_crashes = 10  # very rare (roughly 1% of scenarios)
crash_returns = np.zeros((n_crashes, n_assets))
crash_returns[:, 0] = np.random.normal(-0.55, 0.08, n_crashes)  # SPX: -55% crashes
crash_returns[:, 1] = np.random.normal(-0.05, 0.05, n_crashes)  # Bonds: correlate in crisis!
crash_returns[:, 2] = np.random.normal(0.15, 0.10, n_crashes)   # Gold: modest help
crash_returns[:, 3] = np.random.normal(5.00, 1.00, n_crashes)   # Tail hedge: +500% explosion

# Combine scenarios
returns = np.vstack([normal_returns, crash_returns])
n_scenarios = returns.shape[0]

print("="*60)
print("SCENARIO STATISTICS")
print("="*60)
print(f"Total scenarios: {n_scenarios}")
print(f"Assets: SPX, Bonds, Gold, Tail Hedge")
print(f"\nMean returns per asset:  {returns.mean(axis=0).round(4)}")
print(f"Std dev per asset:       {returns.std(axis=0).round(4)}")

# -----------------------------
# CVaR Optimization (Rockafellar-Uryasev)
# -----------------------------

alpha = 0.05  # 5% tail

# Decision variables
w = cp.Variable(n_assets)      # portfolio weights
zeta = cp.Variable()           # auxiliary variable (becomes VaR at optimum)
u = cp.Variable(n_scenarios)   # shortfall variables

# Portfolio returns for each scenario: R @ w gives (n_scenarios,) vector
portfolio_returns = returns @ w

# Constraints
constraints = [
    u >= 0,
    u >= zeta - portfolio_returns,  # u_i = max(0, zeta - r_i^T w)
    cp.sum(w) == 1,                 # fully invested
    w >= 0,                         # long only
]

# Objective: maximize CVaR
# CVaR = zeta - (1 / (alpha * N)) * sum(u)
objective = cp.Maximize(zeta - (1 / (alpha * n_scenarios)) * cp.sum(u))

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

print("\n" + "="*60)
print("CVaR OPTIMIZATION RESULTS")
print("="*60)
print(f"Status: {problem.status}")
print(f"\nOptimal weights:")
print(f"  SPX:        {w.value[0]:.4f}")
print(f"  Bonds:      {w.value[1]:.4f}")
print(f"  Gold:       {w.value[2]:.4f}")
print(f"  Tail Hedge: {w.value[3]:.4f}")
print(f"\nRisk measures:")
print(f"  VaR (5%):   {zeta.value:.4f}  (5th percentile return)")
print(f"  CVaR (5%):  {problem.value:.4f}  (expected return in worst 5%)")

# -----------------------------
# Verify by direct calculation
# -----------------------------

optimal_portfolio_returns = returns @ w.value
sorted_returns = np.sort(optimal_portfolio_returns)
n_tail = int(alpha * n_scenarios)

var_direct = sorted_returns[n_tail]
cvar_direct = sorted_returns[:n_tail].mean()

print(f"\nDirect calculation verification:")
print(f"  VaR (5%):   {var_direct:.4f}")
print(f"  CVaR (5%):  {cvar_direct:.4f}")

# -----------------------------
# Compare with 100% SPX baseline
# -----------------------------

spx_returns = returns[:, 0]
spx_sorted = np.sort(spx_returns)
spx_var = spx_sorted[n_tail]
spx_cvar = spx_sorted[:n_tail].mean()

print("\n" + "="*60)
print("COMPARISON WITH 100% SPX")
print("="*60)
print(f"{'Metric':<20} {'Optimal':<12} {'100% SPX':<12} {'Improvement':<12}")
print("-"*60)
print(f"{'Mean return':<20} {optimal_portfolio_returns.mean():<12.4f} {spx_returns.mean():<12.4f}")
print(f"{'Std dev':<20} {optimal_portfolio_returns.std():<12.4f} {spx_returns.std():<12.4f}")
print(f"{'VaR (5%)':<20} {zeta.value:<12.4f} {spx_var:<12.4f} {zeta.value - spx_var:<12.4f}")
print(f"{'CVaR (5%)':<20} {problem.value:<12.4f} {spx_cvar:<12.4f} {problem.value - spx_cvar:<12.4f}")

# Approximate CAGR (using log utility)
cagr_optimal = np.mean(np.log(1 + optimal_portfolio_returns))
cagr_spx = np.mean(np.log(1 + spx_returns))
print(f"{'E[log(1+r)]':<20} {cagr_optimal:<12.4f} {cagr_spx:<12.4f} {cagr_optimal - cagr_spx:<12.4f}")

# -----------------------------
# CVaR with minimum return constraint
# -----------------------------

print("\n" + "="*60)
print("CVaR WITH MINIMUM RETURN CONSTRAINT")
print("="*60)

w2 = cp.Variable(n_assets)
zeta2 = cp.Variable()
u2 = cp.Variable(n_scenarios)

portfolio_returns2 = returns @ w2
mean_return2 = cp.sum(portfolio_returns2) / n_scenarios

constraints2 = [
    u2 >= 0,
    u2 >= zeta2 - portfolio_returns2,
    cp.sum(w2) == 1,
    w2 >= 0,
    mean_return2 >= 0.06,  # require at least 6% expected return
]

objective2 = cp.Maximize(zeta2 - (1 / (alpha * n_scenarios)) * cp.sum(u2))
problem2 = cp.Problem(objective2, constraints2)
problem2.solve()

print(f"Status: {problem2.status}")
print(f"\nOptimal weights (with mean >= 6% constraint):")
print(f"  SPX:        {w2.value[0]:.4f}")
print(f"  Bonds:      {w2.value[1]:.4f}")
print(f"  Gold:       {w2.value[2]:.4f}")
print(f"  Tail Hedge: {w2.value[3]:.4f}")

opt_ret2 = returns @ w2.value
print(f"\nMean return:  {opt_ret2.mean():.4f}")
print(f"VaR (5%):     {zeta2.value:.4f}")
print(f"CVaR (5%):    {problem2.value:.4f}")

# -----------------------------
# Kelly Criterion (Max Log Utility)
# -----------------------------

print("\n" + "="*60)
print("KELLY CRITERION (MAXIMIZE LOG UTILITY)")
print("="*60)

w3 = cp.Variable(n_assets)
portfolio_returns3 = returns @ w3

# Maximize expected log return
# Note: need 1 + returns to be positive for log
objective3 = cp.Maximize(cp.sum(cp.log(1 + portfolio_returns3)) / n_scenarios)

constraints3 = [
    cp.sum(w3) == 1,
    w3 >= 0,
    portfolio_returns3 >= -0.99,  # ensure log is defined (no total wipeout)
]

problem3 = cp.Problem(objective3, constraints3)
problem3.solve()

print(f"Status: {problem3.status}")
print(f"\nOptimal weights (Kelly):")
print(f"  SPX:        {w3.value[0]:.4f}")
print(f"  Bonds:      {w3.value[1]:.4f}")
print(f"  Gold:       {w3.value[2]:.4f}")
print(f"  Tail Hedge: {w3.value[3]:.4f}")

opt_ret3 = returns @ w3.value
sorted3 = np.sort(opt_ret3)
print(f"\nMean return:  {opt_ret3.mean():.4f}")
print(f"VaR (5%):     {sorted3[n_tail]:.4f}")
print(f"CVaR (5%):    {sorted3[:n_tail].mean():.4f}")
print(f"E[log(1+r)]:  {np.mean(np.log(1 + opt_ret3)):.4f}")

# -----------------------------
# Summary comparison
# -----------------------------

print("\n" + "="*60)
print("SUMMARY: TAIL HEDGE ALLOCATION BY METHOD")
print("="*60)
print(f"{'Method':<35} {'Tail Hedge %':<15} {'E[log(1+r)]':<12}")
print("-"*60)
print(f"{'CVaR (unconstrained)':<35} {w.value[3]*100:<15.2f} {cagr_optimal:<12.4f}")
print(f"{'CVaR (mean >= 6%)':<35} {w2.value[3]*100:<15.2f} {np.mean(np.log(1+opt_ret2)):<12.4f}")
print(f"{'Kelly (max log utility)':<35} {w3.value[3]*100:<15.2f} {np.mean(np.log(1+opt_ret3)):<12.4f}")
print(f"{'100% SPX':<35} {0:<15.2f} {cagr_spx:<12.4f}")

# -----------------------------
# Kelly Utility Landscape: Sweep tail hedge allocation
# -----------------------------

print("\n" + "="*60)
print("KELLY UTILITY vs TAIL HEDGE ALLOCATION")
print("="*60)
print("Fixing SPX + Tail Hedge, letting optimizer choose rest")
print()

tail_hedge_fractions = np.linspace(0, 0.15, 16)
kelly_utilities = []

for th_frac in tail_hedge_fractions:
    w_sweep = cp.Variable(n_assets)
    port_ret_sweep = returns @ w_sweep
    
    obj_sweep = cp.Maximize(cp.sum(cp.log(1 + port_ret_sweep)) / n_scenarios)
    cons_sweep = [
        cp.sum(w_sweep) == 1,
        w_sweep >= 0,
        w_sweep[3] == th_frac,  # fix tail hedge allocation
        port_ret_sweep >= -0.99,
    ]
    
    prob_sweep = cp.Problem(obj_sweep, cons_sweep)
    prob_sweep.solve()
    
    kelly_utilities.append(prob_sweep.value if prob_sweep.status == 'optimal' else np.nan)

print(f"{'Tail Hedge %':<15} {'E[log(1+r)]':<15} {'vs 0% TH':<15}")
print("-"*45)
baseline_kelly = kelly_utilities[0]
for th_frac, kelly_u in zip(tail_hedge_fractions, kelly_utilities):
    diff = kelly_u - baseline_kelly if not np.isnan(kelly_u) else np.nan
    print(f"{th_frac*100:<15.1f} {kelly_u:<15.4f} {diff:+.4f}")

# Find optimal
best_idx = np.nanargmax(kelly_utilities)
print(f"\nOptimal tail hedge allocation: {tail_hedge_fractions[best_idx]*100:.1f}%")
print(f"Kelly utility improvement over 0%: {kelly_utilities[best_idx] - baseline_kelly:+.6f}")

# -----------------------------
# Two-Asset Analysis: SPX + Tail Hedge Only
# This is the core of Spitznagel's argument
# -----------------------------

print("\n" + "="*60)
print("TWO-ASSET KELLY: SPX + TAIL HEDGE ONLY")
print("="*60)

# Just use SPX and Tail Hedge columns
returns_2asset = returns[:, [0, 3]]

print(f"{'TH Alloc %':<12} {'Mean Ret':<12} {'Std Dev':<12} {'E[log(1+r)]':<12} {'vs 0% TH':<12}")
print("-"*60)

th_allocations = np.linspace(0, 0.10, 21)
results_2asset = []

for th_alloc in th_allocations:
    weights_2a = np.array([1 - th_alloc, th_alloc])
    port_ret_2a = returns_2asset @ weights_2a
    
    mean_ret = port_ret_2a.mean()
    std_ret = port_ret_2a.std()
    
    # Kelly utility (handle potential negative returns carefully)
    valid_returns = port_ret_2a[port_ret_2a > -1]
    if len(valid_returns) == len(port_ret_2a):
        kelly_u = np.mean(np.log(1 + port_ret_2a))
    else:
        kelly_u = np.nan
    
    results_2asset.append((th_alloc, mean_ret, std_ret, kelly_u))

baseline_2a = results_2asset[0][3]
for th_alloc, mean_ret, std_ret, kelly_u in results_2asset:
    diff = kelly_u - baseline_2a if not np.isnan(kelly_u) and not np.isnan(baseline_2a) else np.nan
    print(f"{th_alloc*100:<12.1f} {mean_ret:<12.4f} {std_ret:<12.4f} {kelly_u:<12.4f} {diff:+.4f}")

# Find optimal
kelly_vals_2a = [r[3] for r in results_2asset]
valid_kelly = [(i, k) for i, k in enumerate(kelly_vals_2a) if not np.isnan(k)]
if valid_kelly:
    best_idx_2a = max(valid_kelly, key=lambda x: x[1])[0]
    print(f"\nOptimal tail hedge: {th_allocations[best_idx_2a]*100:.1f}%")
    print(f"Kelly utility at optimum: {kelly_vals_2a[best_idx_2a]:.4f}")

# -----------------------------
# Sensitivity: What crash parameters make tail hedge Kelly-optimal?
# -----------------------------

print("\n" + "="*60)
print("SENSITIVITY: WHEN DOES TAIL HEDGE BECOME KELLY-OPTIMAL?")
print("="*60)
print("Testing different crash severity and tail hedge payoff combinations")
print()

def test_scenario(crash_spx, crash_th, n_crash, normal_th_return):
    """Test if tail hedge is Kelly-optimal for given parameters"""
    np.random.seed(42)
    
    # Normal returns (use stored normal returns for SPX and TH)
    n_normal = 1000
    normal_spx = np.random.normal(0.10, 0.18, n_normal)
    normal_th = np.random.normal(normal_th_return, 0.40, n_normal)
    
    # Crash returns
    crash_spx_ret = np.random.normal(crash_spx, 0.05, n_crash)
    crash_th_ret = np.random.normal(crash_th, crash_th * 0.2, n_crash)
    
    # Combine
    all_spx = np.concatenate([normal_spx, crash_spx_ret])
    all_th = np.concatenate([normal_th, crash_th_ret])
    
    # Test allocations
    best_kelly = -np.inf
    best_alloc = 0
    
    for th_alloc in np.linspace(0, 0.10, 21):
        port_ret = (1 - th_alloc) * all_spx + th_alloc * all_th
        if np.all(port_ret > -1):
            kelly_u = np.mean(np.log(1 + port_ret))
            if kelly_u > best_kelly:
                best_kelly = kelly_u
                best_alloc = th_alloc
    
    return best_alloc, best_kelly

print(f"{'Crash SPX':<12} {'TH Payoff':<12} {'N Crashes':<12} {'TH Bleed':<12} {'Opt TH%':<12}")
print("-"*60)

# Test different scenarios
test_cases = [
    (-0.40, 2.0, 20, -0.15),   # moderate crash, moderate payoff
    (-0.50, 3.0, 15, -0.15),   # severe crash, good payoff
    (-0.50, 5.0, 10, -0.15),   # severe crash, great payoff
    (-0.60, 5.0, 10, -0.15),   # very severe crash
    (-0.60, 8.0, 10, -0.15),   # very severe, huge payoff
    (-0.60, 8.0, 10, -0.10),   # huge payoff, smaller bleed
    (-0.60, 10.0, 10, -0.08),  # extreme payoff, small bleed
]

for crash_spx, crash_th, n_crash, th_bleed in test_cases:
    opt_alloc, opt_kelly = test_scenario(crash_spx, crash_th, n_crash, th_bleed)
    marker = "***" if opt_alloc > 0 else ""
    print(f"{crash_spx:<12.0%} {crash_th:<12.0%} {n_crash:<12} {th_bleed:<12.0%} {opt_alloc*100:<12.1f} {marker}")