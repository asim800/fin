
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for clean output

# Set style for better plots
# plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)  # For reproducibility


def simulate_jump_diffusion(S0, T, mu, sigma, lamb, mu_j, sigma_j, dt=1/252):
    """
    Simulates a single path of the Merton jump-diffusion model.
    
    Parameters:
    S0 (float): Initial stock price
    T (float): Total time in years
    mu (float): Drift coefficient (annual)
    sigma (float): Diffusion volatility (annual)
    lamb (float): Jump intensity (average jumps per year)
    mu_j (float): Mean of the log jump size
    sigma_j (float): Standard deviation of the log jump size
    dt (float): Time step in years (default 1/252 for a trading day)
    
    Returns:
    tuple: (time array, price path array, jump_times)
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    
    # Standard Brownian motion increments
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Poisson process for jumps
    # The probability of a jump in each interval is lambda * dt
    poisson_probs = np.random.rand(n_steps)
    jump_indicators = (poisson_probs < lamb * dt).astype(int)
    num_jumps = np.sum(jump_indicators)
    
    # Generate jump sizes for the times when jumps occur
    jump_sizes = np.exp(mu_j + sigma_j * np.random.randn(num_jumps))
    
    # Track the times when jumps occurred
    jump_times = t[1:][jump_indicators == 1]
    jump_counter = 0
    
    # Simulate the path
    for i in range(1, n_steps + 1):
        # Diffusion component
        S[i] = S[i-1] + mu * S[i-1] * dt + sigma * S[i-1] * dW[i-1]
        
        # Jump component (if a jump occurs at this time step)
        if jump_indicators[i-1]:
            S[i] = S[i] * jump_sizes[jump_counter]
            jump_counter += 1
            
    return t, S, jump_times


def growth_rate(f, mu, sigma, lamb, mu_j, sigma_j, r=0.0):
    """
    Calculates the expected growth rate g(f) for a betting fraction f
    in the Merton jump-diffusion model.
    
    This is the function we want to maximize.
    """
    # Continuous component
    continuous_part = (mu - r) * f - 0.5 * (sigma * f) ** 2
    
    # Jump component: expectation over log(1 + f*(J-1))
    # We approximate the integral using Monte Carlo
    n_samples = 10000
    jump_sizes = np.exp(mu_j + sigma_j * np.random.randn(n_samples))
    jump_returns = f * (jump_sizes - 1)
    # Avoid log(0) for extreme negative jumps
    log_jump_terms = np.log(1 + jump_returns)
    log_jump_terms[jump_returns <= -1] = -np.inf  # Ruin
    
    jump_expectation = np.mean(log_jump_terms)
    
    # Total growth rate
    g = continuous_part + lamb * jump_expectation
    return -g  # Negative because we will minimize



##
# Parameters for the jump-diffusion process
S0 = 100.0       # Initial price
T = 1.0          # 1 year
mu = 0.10        # 10% annual drift
sigma = 0.15     # 15% annual volatility
r = 0.00         # 0% risk-free rate

# Jump parameters - let's create a risky asset!
lamb = 2.0       # 2 jumps per year on average
mu_j = 0 # -0.1      # Average jump size is a 10% drop (exp(-0.1) ≈ 0.905)
sigma_j = 0 # 0.05   # Some variability in jump sizes

# Classic Kelly fraction (IGNORES JUMPS - this is dangerous!)
f_classic = (mu - r) / (sigma ** 2)
print(f"Classic Kelly fraction (ignores jumps): {f_classic:.3f} ({f_classic*100:.1f}% of wealth)")

# Find the optimal fraction f* that MAXIMIZES growth_rate(f)
# We minimize the negative growth rate
result = minimize_scalar(growth_rate, bounds=(0, 1.5), args=(mu, sigma, lamb, mu_j, sigma_j, r), method='bounded')
f_optimal = result.x
print(f"Optimal Kelly fraction (with jumps): {f_optimal:.3f} ({f_optimal*100:.1f}% of wealth)")
print(f"Maximum growth rate g(f*): {-result.fun:.4f}")

# Let's also check what happens if we bet half the optimal amount
f_half = f_optimal / 2
print(f"\nHalf-Kelly fraction: {f_half:.3f} ({f_half*100:.1f}% of wealth)")

##
def simulate_strategy(fraction, n_sims=1000):
    """Simulate the final wealth of a strategy betting a fixed fraction"""
    final_wealth = np.zeros(n_sims)
    
    for i in range(n_sims):
        # Simulate price path
        t, S, _ = simulate_jump_diffusion(S0, T, mu, sigma, lamb, mu_j, sigma_j)
        
        # Calculate strategy returns
        returns = np.diff(S) / S[:-1]  # Simple returns
        strategy_returns = fraction * returns
        
        # Calculate final wealth (avoiding ruin)
        wealth_path = np.cumprod(1 + strategy_returns)
        final_wealth[i] = wealth_path[-1] if np.all(wealth_path > 0) else 0.0
    
    return final_wealth

# Simulate different strategies
n_simulations = 5000
wealth_classic = simulate_strategy(f_classic, n_simulations)
wealth_optimal = simulate_strategy(f_optimal, n_simulations)
wealth_half = simulate_strategy(f_half, n_simulations)

# Calculate performance metrics
def analyze_performance(wealth_values, strategy_name):
    """Calculate and print performance metrics"""
    mean_wealth = np.mean(wealth_values)
    median_wealth = np.median(wealth_values)
    std_wealth = np.std(wealth_values)
    ruin_prob = np.mean(wealth_values <= 0.01)  # Probability of losing 99%+
    
    print(f"\n{strategy_name}:")
    print(f"  Mean final wealth: {mean_wealth:.3f}")
    print(f"  Median final wealth: {median_wealth:.3f}")
    print(f"  Std of final wealth: {std_wealth:.3f}")
    print(f"  Probability of ruin: {ruin_prob:.3f}")
    print(f"  Kelly criterion: log(mean) = {np.log(mean_wealth) if mean_wealth > 0 else -np.inf:.3f}")
    
    return {
        'mean': mean_wealth,
        'median': median_wealth,
        'std': std_wealth,
        'ruin_prob': ruin_prob
    }

# Analyze each strategy
metrics_classic = analyze_performance(wealth_classic, f"Classic Kelly ({f_classic:.2f})")
metrics_optimal = analyze_performance(wealth_optimal, f"Optimal Kelly ({f_optimal:.2f})")
metrics_half = analyze_performance(wealth_half, f"Half-Kelly ({f_half:.2f})")

##
# Create a comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, figure=fig)

# Plot 1: Sample price paths with jumps
ax1 = fig.add_subplot(gs[0, :])
for _ in range(5):
    t, S, jump_times = simulate_jump_diffusion(S0, T, mu, sigma, lamb, mu_j, sigma_j)
    ax1.plot(t, S, alpha=0.7)
    if len(jump_times) > 0:
        jump_prices = np.interp(jump_times, t, S)
        ax1.scatter(jump_times, jump_prices, color='red', s=30, zorder=5)
ax1.set_title('Sample Jump-Diffusion Price Paths (Red dots = jumps)')
ax1.set_ylabel('Price')
ax1.grid(True, alpha=0.3)

# Plot 2: Growth rate function
ax2 = fig.add_subplot(gs[1, 0])
f_values = np.linspace(0, 1.5, 100)
growth_rates = [-growth_rate(f, mu, sigma, lamb, mu_j, sigma_j, r) for f in f_values]
ax2.plot(f_values, growth_rates, linewidth=2)
ax2.axvline(f_classic, color='red', linestyle='--', alpha=0.7, label=f'Classic Kelly: {f_classic:.2f}')
ax2.axvline(f_optimal, color='green', linestyle='--', alpha=0.7, label=f'Optimal Kelly: {f_optimal:.2f}')
ax2.set_xlabel('Betting Fraction (f)')
ax2.set_ylabel('Expected Growth Rate g(f)')
ax2.set_title('Growth Rate vs. Betting Fraction')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of jump sizes
ax3 = fig.add_subplot(gs[1, 1])
jump_samples = np.exp(mu_j + sigma_j * np.random.randn(10000))
ax3.hist(jump_samples, bins=50, density=True, alpha=0.7)
ax3.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='No change (J=1)')
ax3.axvline(np.exp(mu_j), color='green', linestyle='--', alpha=0.7, label=f'Mean: {np.exp(mu_j):.2f}')
ax3.set_xlabel('Jump Size (J)')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Jump Sizes')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Final wealth distributions
ax4 = fig.add_subplot(gs[2, :])
# Use logarithmic scale for better visualization
bins = np.logspace(-3, 3, 100)
ax4.hist(wealth_classic, bins=bins, alpha=0.5, label=f'Classic Kelly ({f_classic:.2f})', density=True)
ax4.hist(wealth_optimal, bins=bins, alpha=0.5, label=f'Optimal Kelly ({f_optimal:.2f})', density=True)
ax4.hist(wealth_half, bins=bins, alpha=0.5, label=f'Half-Kelly ({f_half:.2f})', density=True)
ax4.set_xscale('log')
ax4.set_xlabel('Final Wealth (Log Scale)')
ax4.set_ylabel('Density')
ax4.set_title('Distribution of Final Wealth (Log Scale)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"{'Strategy':<20} {'Mean Wealth':<12} {'Ruin Prob':<10} {'Kelly Criterion'}")
print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*15}")
for name, metrics in [
    (f"Classic ({f_classic:.2f})", metrics_classic),
    (f"Optimal ({f_optimal:.2f})", metrics_optimal),
    (f"Half ({f_half:.2f})", metrics_half)
]:
    kelly_val = np.log(metrics['mean']) if metrics['mean'] > 0 else -np.inf
    print(f"{name:<20} {metrics['mean']:<12.3f} {metrics['ruin_prob']:<10.3f} {kelly_val:.3f}")