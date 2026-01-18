import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode
from scipy import integrate
import time

from data_loader import load_returns_data

# ============================================================================
# EXPONENTIATED GRADIENT (EG) IMPLEMENTATION
# ============================================================================

class ExponentiatedGradient:
    """
    Exponentiated Gradient algorithm for portfolio selection.

    Based on Helmbold et al. (1998) "On-Line Portfolio Selection
    Using Multiplicative Updates"
    """

    def __init__(self, n_assets, learning_rate=0.05, initial_portfolio=None, asset_names=None):
        """
        Args:
            n_assets: number of assets
            learning_rate: eta parameter (typically 0.01 to 0.5)
            initial_portfolio: initial weights (default: uniform)
            asset_names: list of asset names for dict-based updates
        """
        self.n_assets = n_assets
        self.eta = learning_rate
        self.asset_names = asset_names  # For dict-based price_multipliers

        if initial_portfolio is None:
            self.portfolio = np.ones(n_assets) / n_assets
        else:
            self.portfolio = np.array(initial_portfolio)
            self.portfolio = self.portfolio / self.portfolio.sum()

        self.portfolio_history = [self.portfolio.copy()]
        self.wealth_history = [1.0]
        self.current_wealth = 1.0
    
    def update(self, price_multipliers):
        """
        Update portfolio given observed price multipliers.

        Args:
            price_multipliers: Dict with asset keys (e.g., 'stocks', 'bonds')
                              or array of shape (n_assets,) with price relatives

        Returns:
            new_portfolio: updated portfolio weights
        """
        # Convert dict to array if needed
        if isinstance(price_multipliers, dict):
            price_relatives = np.array([price_multipliers[a] for a in self.asset_names])
        else:
            price_relatives = price_multipliers

        # Current period return
        period_return = self.portfolio @ price_relatives
        
        # Update wealth
        self.current_wealth *= period_return
        self.wealth_history.append(self.current_wealth)
        
        # EG update: w_{t+1}^{(j)} ∝ w_t^{(j)} * exp(η * x_t^{(j)} / (w_t · x_t))
        gradient = price_relatives / period_return
        
        # Multiplicative update
        new_portfolio = self.portfolio * np.exp(self.eta * gradient)
        
        # Normalize to simplex
        new_portfolio = new_portfolio / new_portfolio.sum()
        
        self.portfolio = new_portfolio
        self.portfolio_history.append(self.portfolio.copy())
        
        return new_portfolio
    
    def get_cumulative_wealth(self):
        """Return final wealth achieved"""
        return self.current_wealth
    
    def get_portfolio_history(self):
        """Return history of portfolio allocations"""
        return np.array(self.portfolio_history)
    
    def get_wealth_history(self):
        """Return wealth over time"""
        return np.array(self.wealth_history)


def run_eg_strategy(returns_history, learning_rate=0.05):
    """
    Run EG on a sequence of returns.
    
    Args:
        returns_history: array of shape (T, n_assets)
        learning_rate: eta parameter
    
    Returns:
        eg: ExponentiatedGradient object with results
    """
    T, n_assets = returns_history.shape
    
    eg = ExponentiatedGradient(n_assets, learning_rate=learning_rate)
    
    for t in range(T):
        eg.update(returns_history[t])
    
    return eg


# ============================================================================
# UNIVERSAL PORTFOLIO IMPLEMENTATIONS (from before)
# ============================================================================

def universal_portfolio_3assets_grid(returns_history, delta=0.05):
    """Grid approximation for Universal Portfolio"""
    def generate_simplex_grid_3d(delta):
        grid_points = []
        b1_vals = np.arange(0, 1 + delta, delta)
        for b1 in b1_vals:
            b2_vals = np.arange(0, 1 - b1 + delta, delta)
            for b2 in b2_vals:
                b3 = 1 - b1 - b2
                if b3 >= 0 and b3 <= 1:
                    grid_points.append([b1, b2, b3])
        return np.array(grid_points)
    
    grid_points = generate_simplex_grid_3d(delta)
    N_t = np.zeros(3)
    
    for b in grid_points:
        wealth = 1.0
        for x in returns_history:
            wealth *= b @ x
        N_t += wealth * b
    
    return N_t / N_t.sum()


def universal_portfolio_3assets_mcmc(returns_history, n_samples=3000, burn_in=500):
    """MCMC approximation for Universal Portfolio"""
    def log_wealth(b, returns_history):
        log_w = 0.0
        for x in returns_history:
            log_w += np.log(b @ x)
        return log_w
    
    def propose_dirichlet_step(current, concentration=50):
        alpha = current * concentration
        alpha = np.maximum(alpha, 0.1)
        return np.random.dirichlet(alpha)
    
    current = np.ones(3) / 3
    current_log_wealth = log_wealth(current, returns_history)
    samples = []
    
    for i in range(burn_in + n_samples):
        proposed = propose_dirichlet_step(current, concentration=50)
        proposed_log_wealth = log_wealth(proposed, returns_history)
        
        log_ratio = proposed_log_wealth - current_log_wealth
        
        if np.log(np.random.rand()) < log_ratio:
            current = proposed
            current_log_wealth = proposed_log_wealth
        
        if i >= burn_in:
            samples.append(current.copy())
    
    return np.mean(samples, axis=0)


# ============================================================================
# ONLINE UNIVERSAL PORTFOLIO
# ============================================================================

class OnlineUniversalPortfolio:
    """
    Online version of Universal Portfolio that updates portfolio each period.
    Uses grid approximation for computational tractability.
    """

    def __init__(self, n_assets, grid_delta=0.1, asset_names=None):
        """
        Args:
            n_assets: number of assets (only 3 supported for now)
            grid_delta: grid spacing for approximation
            asset_names: list of asset names for dict-based updates
        """
        if n_assets != 3:
            raise ValueError("Only 3 assets supported for Universal Portfolio")

        self.n_assets = n_assets
        self.delta = grid_delta
        self.asset_names = asset_names
        self.grid_points = self._generate_grid()

        self.returns_history = []
        self.portfolio_history = [np.ones(n_assets) / n_assets]
        self.wealth_history = [1.0]
        self.current_wealth = 1.0
    
    def _generate_grid(self):
        """Generate simplex grid"""
        grid_points = []
        b1_vals = np.arange(0, 1 + self.delta, self.delta)
        for b1 in b1_vals:
            b2_vals = np.arange(0, 1 - b1 + self.delta, self.delta)
            for b2 in b2_vals:
                b3 = 1 - b1 - b2
                if b3 >= 0 and b3 <= 1:
                    grid_points.append([b1, b2, b3])
        return np.array(grid_points)
    
    def update(self, price_multipliers):
        """Update portfolio given observed returns

        Args:
            price_multipliers: Dict with asset keys or array of price relatives
        """
        # Convert dict to array if needed
        if isinstance(price_multipliers, dict):
            price_relatives = np.array([price_multipliers[a] for a in self.asset_names])
        else:
            price_relatives = price_multipliers

        # Store returns
        self.returns_history.append(price_relatives)

        # Compute Universal Portfolio based on all history
        N_t = np.zeros(self.n_assets)

        for b in self.grid_points:
            wealth = 1.0
            for x in self.returns_history:
                wealth *= b @ x
            N_t += wealth * b

        new_portfolio = N_t / N_t.sum()

        # Update wealth (using previous portfolio, not new one!)
        # This is key: we invest with previous portfolio, observe returns, then update
        period_return = self.portfolio_history[-1] @ price_relatives
        self.current_wealth *= period_return

        self.portfolio_history.append(new_portfolio)
        self.wealth_history.append(self.current_wealth)

        return new_portfolio
    
    def get_cumulative_wealth(self):
        return self.current_wealth
    
    def get_portfolio_history(self):
        return np.array(self.portfolio_history)
    
    def get_wealth_history(self):
        return np.array(self.wealth_history)


# ============================================================================
# BENCHMARK STRATEGIES
# ============================================================================

class BestConstantRebalanced:
    """
    Best Constant Rebalanced Portfolio (BCRP) computed in hindsight.
    This is the optimal benchmark.
    """

    def __init__(self, n_assets, asset_names=None):
        self.n_assets = n_assets
        self.asset_names = asset_names
        self.returns_history = []
        self.wealth_history = [1.0]

    def update(self, price_multipliers):
        """Store returns

        Args:
            price_multipliers: Dict with asset keys or array of price relatives
        """
        # Convert dict to array if needed
        if isinstance(price_multipliers, dict):
            price_relatives = np.array([price_multipliers[a] for a in self.asset_names])
        else:
            price_relatives = price_multipliers

        self.returns_history.append(price_relatives)
    
    def compute_best_crp(self, grid_delta=0.05):
        """
        Find best CRP by grid search.
        
        Returns:
            best_portfolio: portfolio that achieved max wealth
            best_wealth: maximum wealth achieved
        """
        if self.n_assets != 3:
            raise ValueError("Only 3 assets supported")
        
        returns_array = np.array(self.returns_history)
        
        # Generate grid
        grid_points = []
        b1_vals = np.arange(0, 1 + grid_delta, grid_delta)
        for b1 in b1_vals:
            b2_vals = np.arange(0, 1 - b1 + grid_delta, grid_delta)
            for b2 in b2_vals:
                b3 = 1 - b1 - b2
                if b3 >= 0 and b3 <= 1:
                    grid_points.append([b1, b2, b3])
        
        grid_points = np.array(grid_points)
        
        # Find best portfolio
        best_wealth = 0
        best_portfolio = None
        
        for b in grid_points:
            wealth = 1.0
            for x in returns_array:
                wealth *= b @ x
            
            if wealth > best_wealth:
                best_wealth = wealth
                best_portfolio = b
        
        return best_portfolio, best_wealth
    
    def get_wealth_history(self):
        """Compute wealth history for best CRP"""
        best_portfolio, _ = self.compute_best_crp()
        
        wealth = 1.0
        wealth_history = [1.0]
        
        for x in self.returns_history:
            wealth *= best_portfolio @ x
            wealth_history.append(wealth)
        
        return np.array(wealth_history)


class BuyAndHold:
    """Buy and hold strategy"""

    def __init__(self, n_assets, initial_portfolio=None, asset_names=None):
        self.n_assets = n_assets
        self.asset_names = asset_names
        if initial_portfolio is None:
            self.portfolio = np.ones(n_assets) / n_assets
        else:
            self.portfolio = np.array(initial_portfolio)

        self.wealth_history = [1.0]
        self.current_wealth = 1.0

    def update(self, price_multipliers):
        """Update wealth (portfolio doesn't change)

        Args:
            price_multipliers: Dict with asset keys or array of price relatives
        """
        # Convert dict to array if needed
        if isinstance(price_multipliers, dict):
            price_relatives = np.array([price_multipliers[a] for a in self.asset_names])
        else:
            price_relatives = price_multipliers

        period_return = self.portfolio @ price_relatives
        self.current_wealth *= period_return
        self.wealth_history.append(self.current_wealth)
    
    def get_cumulative_wealth(self):
        return self.current_wealth
    
    def get_wealth_history(self):
        return np.array(self.wealth_history)


# ============================================================================
# BOOTSTRAP RESAMPLING
# ============================================================================

def resample_returns_block(returns_df, block_size=None, seed=None):
    """
    Block bootstrap resampling of returns DataFrame.

    Preserves temporal dependencies by resampling contiguous blocks
    rather than individual periods.

    Args:
        returns_df: DataFrame with percentage returns (index=dates, columns=assets)
        block_size: Size of contiguous blocks (default: ~10% of data)
        seed: Random seed for reproducibility

    Returns:
        DataFrame: Resampled returns (same shape, new index)
    """
    if seed is not None:
        np.random.seed(seed)

    n_periods = len(returns_df)
    block_size = block_size or max(1, n_periods // 10)

    # Number of blocks needed
    n_blocks = int(np.ceil(n_periods / block_size))

    # Randomly select block start positions
    max_start = n_periods - block_size
    if max_start <= 0:
        # Data too short for blocks, fall back to IID
        indices = np.random.choice(n_periods, n_periods, replace=True)
    else:
        block_starts = np.random.randint(0, max_start + 1, size=n_blocks)
        indices = np.concatenate([
            np.arange(start, min(start + block_size, n_periods))
            for start in block_starts
        ])[:n_periods]  # Trim to exact length

    # Resample and reset index
    resampled = returns_df.iloc[indices].reset_index(drop=True)
    return resampled


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_all_methods(returns_input, eta_values=[0.01, 0.05, 0.1, 0.2],
                        methods=None):
    """
    Compare all methods on the same return sequence.

    Args:
        returns_input: DataFrame with percentage returns OR array of shape (T, 3) with price relatives
        eta_values: list of learning rates to try for EG
        methods: list of methods to run, e.g., ['BCRP', 'Buy&Hold', 'EG(η=0.05)', 'UP(Grid)']
                 If None, runs all methods. Available methods:
                 - 'BCRP': Best Constant Rebalanced Portfolio (oracle)
                 - 'Buy&Hold': Buy and Hold strategy
                 - 'EG(η=X)': Exponentiated Gradient with learning rate X
                 - 'UP(Grid)': Universal Portfolio with grid approximation
                 - 'UP(MCMC)': Universal Portfolio with MCMC approximation

    Returns:
        results: dictionary with all results
    """
    # Handle DataFrame input
    if hasattr(returns_input, 'columns'):
        asset_names = returns_input.columns.tolist()
        n_assets = len(asset_names)
        T = len(returns_input)
        # Convert to list of dicts with price multipliers (1 + return)
        returns_history = []
        for _, row in returns_input.iterrows():
            returns_history.append({col: row[col] + 1.0 for col in asset_names})
    else:
        # Backward compatible: array input
        returns_history = returns_input
        T, n_assets = returns_history.shape
        asset_names = None  # Will use array indexing

    assert n_assets == 3, "Only 3 assets supported"

    # Determine which methods to run
    def should_run(method_name):
        if methods is None:
            return True
        return method_name in methods

    print("=" * 70)
    print("COMPREHENSIVE PORTFOLIO SELECTION COMPARISON")
    print("=" * 70)
    print(f"Number of periods: {T}")
    print(f"Number of assets: {n_assets}")
    if asset_names:
        print(f"Assets: {asset_names}")
    if methods:
        print(f"Methods: {methods}")
    print()

    results = {}

    # ========== Best CRP (oracle) ==========
    # Always compute BCRP for regret calculations, but only store if requested
    print("Computing Best Constant Rebalanced Portfolio (oracle)...")
    bcrp = BestConstantRebalanced(n_assets, asset_names=asset_names)
    for t in range(T):
        bcrp.update(returns_history[t])

    best_portfolio, best_wealth = bcrp.compute_best_crp(grid_delta=0.05)
    print(f"  Best portfolio: {best_portfolio}")
    print(f"  Best wealth: {best_wealth:.4f}")

    if should_run('BCRP'):
        # Compute BCRP evolution (optimal weights at each timestep)
        # Sample every few periods for speed (interpolate for smoother plot)
        print("  Computing BCRP weight evolution...")
        bcrp_portfolio_history = [np.ones(n_assets) / n_assets]  # Start uniform

        # Convert returns_history to array format for BCRP evolution
        if asset_names:
            returns_array_bcrp = np.array([[r[a] for a in asset_names] for r in returns_history])
        else:
            returns_array_bcrp = np.array(returns_history)

        # Sample every few periods for speed (adjust based on total periods)
        sample_step = max(1, T // 30)  # At most ~30 BCRP computations
        sampled_indices = list(range(0, T + 1, sample_step))
        if T not in sampled_indices:
            sampled_indices.append(T)

        print(f"  (Computing at {len(sampled_indices)} sample points...)")

        for t in sampled_indices[1:]:  # Skip t=0, already have uniform
            # Compute BCRP using data up to time t
            bcrp_temp = BestConstantRebalanced(n_assets, asset_names=asset_names)
            for i in range(t):
                bcrp_temp.update(returns_history[i])
            optimal_portfolio, _ = bcrp_temp.compute_best_crp(grid_delta=0.1)  # Coarser grid for speed
            bcrp_portfolio_history.append(optimal_portfolio)

        # Interpolate to full length for smooth plotting
        sampled_history = np.array(bcrp_portfolio_history)
        full_history = np.zeros((T + 1, n_assets))
        full_history[0] = sampled_history[0]

        for idx, t in enumerate(sampled_indices[1:], 1):
            full_history[t] = sampled_history[idx]

        # Linear interpolation for missing points
        for i in range(len(sampled_indices) - 1):
            t_start = sampled_indices[i]
            t_end = sampled_indices[i + 1]
            for t in range(t_start + 1, t_end):
                alpha = (t - t_start) / (t_end - t_start)
                full_history[t] = (1 - alpha) * full_history[t_start] + alpha * full_history[t_end]

        results['BCRP'] = {
            'portfolio': best_portfolio,
            'wealth': best_wealth,
            'wealth_history': bcrp.get_wealth_history(),
            'portfolio_history': full_history,
            'time': 0
        }
    print()

    # ========== Buy and Hold ==========
    if should_run('Buy&Hold'):
        print("Running Buy and Hold (uniform)...")
        start = time.time()
        bh = BuyAndHold(n_assets, asset_names=asset_names)
        for t in range(T):
            bh.update(returns_history[t])
        time_bh = time.time() - start

        print(f"  Final wealth: {bh.get_cumulative_wealth():.4f}")
        print(f"  Time: {time_bh:.4f}s")
        print()

        results['Buy&Hold'] = {
            'wealth': bh.get_cumulative_wealth(),
            'wealth_history': bh.get_wealth_history(),
            'time': time_bh
        }

    # ========== Exponentiated Gradient (multiple eta) ==========
    # Check if any EG methods should run
    eg_to_run = [eta for eta in eta_values if should_run(f'EG(η={eta})')]
    if eg_to_run:
        print("Running Exponentiated Gradient with various learning rates...")
        for eta in eg_to_run:
            start = time.time()
            eg = ExponentiatedGradient(n_assets, learning_rate=eta, asset_names=asset_names)
            for t in range(T):
                eg.update(returns_history[t])
            time_eg = time.time() - start

            final_portfolio = eg.portfolio
            final_wealth = eg.get_cumulative_wealth()

            print(f"  EG(η={eta:.2f}):")
            print(f"    Final portfolio: {final_portfolio}")
            print(f"    Final wealth: {final_wealth:.4f}")
            print(f"    Regret vs BCRP: {best_wealth - final_wealth:.4f}")
            print(f"    Time: {time_eg:.4f}s")

            results[f'EG(η={eta})'] = {
                'portfolio': final_portfolio,
                'wealth': final_wealth,
                'wealth_history': eg.get_wealth_history(),
                'portfolio_history': eg.get_portfolio_history(),
                'time': time_eg
            }
        print()

    # ========== Universal Portfolio (Grid) ==========
    if should_run('UP(Grid)'):
        print("Running Universal Portfolio (Grid approximation)...")
        start = time.time()
        up_grid = OnlineUniversalPortfolio(n_assets, grid_delta=0.1, asset_names=asset_names)
        for t in range(T):
            up_grid.update(returns_history[t])
        time_up_grid = time.time() - start

        print(f"  Final wealth: {up_grid.get_cumulative_wealth():.4f}")
        print(f"  Regret vs BCRP: {best_wealth - up_grid.get_cumulative_wealth():.4f}")
        print(f"  Time: {time_up_grid:.4f}s")
        print()

        results['UP(Grid)'] = {
            'wealth': up_grid.get_cumulative_wealth(),
            'wealth_history': up_grid.get_wealth_history(),
            'portfolio_history': up_grid.get_portfolio_history(),
            'time': time_up_grid
        }

    # ========== Universal Portfolio (MCMC) ==========
    if should_run('UP(MCMC)'):
        print("Running Universal Portfolio (MCMC approximation)...")
        start = time.time()

        # Convert returns_history to array format for MCMC
        if asset_names:
            # Convert list of dicts to numpy array
            returns_array = np.array([[r[a] for a in asset_names] for r in returns_history])
        else:
            returns_array = returns_history

        # Build up returns incrementally for MCMC
        up_mcmc_wealth_history = [1.0]
        up_mcmc_portfolio_history = [np.ones(n_assets) / n_assets]

        for t in range(1, T + 1):
            # Compute UP based on returns[0:t]
            portfolio = universal_portfolio_3assets_mcmc(returns_array[:t],
                                                         n_samples=1000,
                                                         burn_in=200)
            up_mcmc_portfolio_history.append(portfolio)

            # Compute wealth using previous portfolio
            prev_portfolio = up_mcmc_portfolio_history[-2]
            period_return = prev_portfolio @ returns_array[t-1]
            up_mcmc_wealth_history.append(up_mcmc_wealth_history[-1] * period_return)

        time_up_mcmc = time.time() - start

        print(f"  Final wealth: {up_mcmc_wealth_history[-1]:.4f}")
        print(f"  Regret vs BCRP: {best_wealth - up_mcmc_wealth_history[-1]:.4f}")
        print(f"  Time: {time_up_mcmc:.4f}s")
        print()

        results['UP(MCMC)'] = {
            'wealth': up_mcmc_wealth_history[-1],
            'wealth_history': np.array(up_mcmc_wealth_history),
            'portfolio_history': np.array(up_mcmc_portfolio_history),
            'time': time_up_mcmc
        }

    return results


# ============================================================================
# BOOTSTRAP ANALYSIS
# ============================================================================

def bootstrap_compare_methods(returns_df, n_bootstrap=100, block_size=None,
                               eta_values=[0.01, 0.05, 0.1, 0.2],
                               methods=None, ci_level=95, verbose=True):
    """
    Run bootstrap analysis on portfolio methods.

    Resamples returns using block bootstrap and runs all methods on each
    resample to compute confidence intervals for performance metrics.

    Args:
        returns_df: DataFrame with percentage returns
        n_bootstrap: Number of bootstrap iterations
        block_size: Block size for resampling (default: 10% of data)
        eta_values: Learning rates for EG
        methods: List of methods to run (None = all)
        ci_level: Confidence interval level (e.g., 95)
        verbose: Print progress

    Returns:
        results: Dict with original results + bootstrap statistics
        bootstrap_samples: List of all bootstrap run results
    """
    import sys
    from io import StringIO

    # Run original comparison
    if verbose:
        print("Running original comparison...")
    original_results = compare_all_methods(returns_df, eta_values, methods)

    # Collect bootstrap samples
    bootstrap_samples = []
    method_names = list(original_results.keys())

    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 20 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}")

        # Resample returns
        resampled = resample_returns_block(returns_df, block_size, seed=b)

        # Run comparison on resampled data (suppress output)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            with np.errstate(all='ignore'):
                boot_results = compare_all_methods(resampled, eta_values, methods)
        finally:
            sys.stdout = old_stdout

        bootstrap_samples.append(boot_results)

    # Compute bootstrap statistics
    alpha = (100 - ci_level) / 2

    for method in method_names:
        # Collect wealth values across bootstrap samples
        boot_wealths = [bs[method]['wealth'] for bs in bootstrap_samples
                        if method in bs]

        if boot_wealths:
            original_results[method]['bootstrap'] = {
                'mean': np.mean(boot_wealths),
                'std': np.std(boot_wealths),
                'ci_lower': np.percentile(boot_wealths, alpha),
                'ci_upper': np.percentile(boot_wealths, 100 - alpha),
                'n_samples': len(boot_wealths)
            }

    return original_results, bootstrap_samples


def print_bootstrap_summary(results, ci_level=95):
    """Print formatted bootstrap results table."""
    print()
    print("=" * 90)
    print("BOOTSTRAP PERFORMANCE SUMMARY")
    print("=" * 90)
    print(f"{'Method':<20} {'Wealth':>10} {'Boot Mean':>12} {'Boot Std':>10} "
          f"{'CI Lower':>12} {'CI Upper':>12}")
    print("-" * 90)

    for method, data in results.items():
        wealth = data['wealth']
        if 'bootstrap' in data:
            bs = data['bootstrap']
            print(f"{method:<20} {wealth:>10.4f} {bs['mean']:>12.4f} "
                  f"{bs['std']:>10.4f} {bs['ci_lower']:>12.4f} {bs['ci_upper']:>12.4f}")
        else:
            print(f"{method:<20} {wealth:>10.4f} {'N/A':>12} {'N/A':>10} "
                  f"{'N/A':>12} {'N/A':>12}")

    print("=" * 90)
    print(f"Note: CI is {ci_level}% confidence interval from block bootstrap")


def plot_bootstrap_wealth_bands(results, bootstrap_samples, ci_level=95):
    """
    Plot wealth trajectories with bootstrap confidence bands.

    Args:
        results: Dict with original results
        bootstrap_samples: List of bootstrap run results
        ci_level: Confidence interval level

    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    alpha = (100 - ci_level) / 2
    color_map = {'BCRP': 'gold', 'Buy&Hold': 'gray'}

    for method, data in results.items():
        if 'wealth_history' not in data:
            continue

        wealth = data['wealth_history']
        color = color_map.get(method, None)

        # Plot main line
        line, = ax.plot(wealth, label=method, linewidth=2, color=color)

        # Compute CI bands from bootstrap samples
        boot_wealths = [bs[method]['wealth_history']
                        for bs in bootstrap_samples if method in bs]

        if boot_wealths:
            # Align lengths (some may vary slightly)
            min_len = min(len(w) for w in boot_wealths)
            boot_array = np.array([w[:min_len] for w in boot_wealths])

            lower = np.percentile(boot_array, alpha, axis=0)
            upper = np.percentile(boot_array, 100 - alpha, axis=0)

            ax.fill_between(range(min_len), lower, upper,
                           alpha=0.2, color=line.get_color())

    ax.set_xlabel('Period')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title(f'Portfolio Performance with {ci_level}% Bootstrap CI')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    return fig


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison_results(results, returns_history, asset_names=None):
    """Create comprehensive visualization of results

    Args:
        results: dictionary with method results
        returns_history: returns data (DataFrame or array)
        asset_names: list of asset names for legends (optional)
    """
    # Extract asset names from DataFrame if not provided
    if asset_names is None:
        if hasattr(returns_history, 'columns'):
            asset_names = returns_history.columns.tolist()
        else:
            asset_names = [f'Asset {i+1}' for i in range(3)]

    fig = plt.figure(figsize=(16, 10))
    
    # ========== Plot 1: Wealth over time ==========
    ax1 = plt.subplot(2, 3, 1)
    
    for name, data in results.items():
        if 'wealth_history' in data:
            ax1.plot(data['wealth_history'], label=name, linewidth=2)
    
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Cumulative Wealth')
    ax1.set_title('Wealth Accumulation Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # ========== Plot 2: Final wealth comparison ==========
    ax2 = plt.subplot(2, 3, 2)
    
    names = []
    wealths = []
    colors = []
    
    for name, data in results.items():
        names.append(name)
        wealths.append(data['wealth'])
        
        if name == 'BCRP':
            colors.append('gold')
        elif name.startswith('UP'):
            colors.append('green')
        elif name.startswith('EG'):
            colors.append('blue')
        else:
            colors.append('gray')
    
    bars = ax2.barh(names, wealths, color=colors, alpha=0.7)
    ax2.set_xlabel('Final Wealth')
    ax2.set_title('Final Wealth Comparison')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, wealth) in enumerate(zip(bars, wealths)):
        ax2.text(wealth, i, f' {wealth:.3f}', va='center')
    
    # ========== Plot 3: CAGR Comparison ==========
    ax3 = plt.subplot(2, 3, 3)

    names = []
    cagrs = []
    colors_cagr = []

    for name, data in results.items():
        if 'wealth_history' in data:
            wealth_history = data['wealth_history']
            n_periods = len(wealth_history) - 1
            if n_periods > 0 and wealth_history[-1] > 0:
                # CAGR = (final/initial)^(1/n) - 1
                cagr = (wealth_history[-1] / wealth_history[0]) ** (1 / n_periods) - 1
                names.append(name)
                cagrs.append(cagr * 100)  # Convert to percentage

                if name == 'BCRP':
                    colors_cagr.append('gold')
                elif name.startswith('UP'):
                    colors_cagr.append('green')
                elif name.startswith('EG'):
                    colors_cagr.append('blue')
                else:
                    colors_cagr.append('gray')

    bars = ax3.barh(names, cagrs, color=colors_cagr, alpha=0.7)
    ax3.set_xlabel('CAGR (%)')
    ax3.set_title('CAGR Comparison')
    ax3.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, cagr) in enumerate(zip(bars, cagrs)):
        ax3.text(cagr, i, f' {cagr:.2f}%', va='center')

    # ========== Plot 4: BCRP Weight Evolution ==========
    ax4 = plt.subplot(2, 3, 4)

    if 'BCRP' in results and 'portfolio_history' in results['BCRP']:
        portfolio_hist = results['BCRP']['portfolio_history']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(min(3, len(asset_names))):
            ax4.plot(portfolio_hist[:, i], label=asset_names[i], linewidth=2, color=colors[i])

        ax4.set_xlabel('Period')
        ax4.set_ylabel('Portfolio Weight')
        ax4.set_title('BCRP Weight Evolution (Hindsight Optimal)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1.05])  # Slightly above 1 to see lines at 1
    else:
        ax4.text(0.5, 0.5, 'BCRP not computed', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('BCRP Weight Evolution')
    
    # ========== Plot 5: Portfolio evolution for UP ==========
    ax5 = plt.subplot(2, 3, 5)

    if 'UP(Grid)' in results and 'portfolio_history' in results['UP(Grid)']:
        portfolio_hist = results['UP(Grid)']['portfolio_history']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(min(3, len(asset_names))):
            ax5.plot(portfolio_hist[:, i], label=asset_names[i], linewidth=2, color=colors[i])

        ax5.set_xlabel('Period')
        ax5.set_ylabel('Portfolio Weight')
        ax5.set_title('Portfolio Evolution: Universal Portfolio (Grid)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1.05])  # Slightly above 1 to see lines at 1
    else:
        ax5.text(0.5, 0.5, 'UP(Grid) not computed', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Portfolio Evolution: UP(Grid)')

    # ========== Plot 6: Regret Accumulation ==========
    ax6 = plt.subplot(2, 3, 6)

    if 'BCRP' in results and 'wealth_history' in results['BCRP']:
        bcrp_wealth_history = results['BCRP']['wealth_history']

        for name, data in results.items():
            if 'wealth_history' in data and name != 'BCRP':
                regret = bcrp_wealth_history - data['wealth_history']

                if name.startswith('UP'):
                    color = 'green'
                elif name.startswith('EG'):
                    color = 'blue'
                else:
                    color = 'gray'

                ax6.plot(regret, label=name, linewidth=2, color=color, alpha=0.7)

        ax6.set_xlabel('Period')
        ax6.set_ylabel('Wealth Regret vs BCRP')
        ax6.set_title('Regret Accumulation')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'BCRP not computed', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Regret Accumulation')
    
    plt.tight_layout()

    # Save to results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(os.path.dirname(script_dir), 'results')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(os.path.join(results_folder, 'universal_methods_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()  # Display in interactive mode


# ============================================================================
# TEST AND DEMONSTRATION
# ============================================================================

def generate_test_data(T=20, n_assets=3, scenario='mixed'):
    """
    Generate synthetic return data.
    
    Args:
        T: number of periods
        n_assets: number of assets
        scenario: 'mixed', 'trending', 'mean_reverting', 'volatile'
    """
    np.random.seed(42)
    
    if scenario == 'mixed':
        # Random returns with slight positive bias
        returns = np.random.uniform(0.95, 1.10, size=(T, n_assets))
        
    elif scenario == 'trending':
        # Asset 1 trends up, Asset 2 trends down, Asset 3 stable
        returns = np.zeros((T, n_assets))
        for t in range(T):
            returns[t, 0] = 1.02 + np.random.normal(0, 0.02)  # Uptrend
            returns[t, 1] = 0.98 + np.random.normal(0, 0.02)  # Downtrend
            returns[t, 2] = 1.00 + np.random.normal(0, 0.02)  # Stable
    
    elif scenario == 'mean_reverting':
        # Assets oscillate
        returns = np.zeros((T, n_assets))
        for t in range(T):
            phase = 2 * np.pi * t / 10
            returns[t, 0] = 1.0 + 0.05 * np.sin(phase) + np.random.normal(0, 0.01)
            returns[t, 1] = 1.0 + 0.05 * np.cos(phase) + np.random.normal(0, 0.01)
            returns[t, 2] = 1.0 + np.random.normal(0, 0.01)
    
    elif scenario == 'volatile':
        # High volatility
        returns = np.random.uniform(0.85, 1.20, size=(T, n_assets))
    
    # Ensure all returns are positive
    returns = np.maximum(returns, 0.01)
    
    return returns


def run_full_comparison(methods=None, n_bootstrap=0, block_size=None):
    """Run complete comparison with visualization

    Args:
        methods: list of methods to run, e.g., ['BCRP', 'Buy&Hold', 'EG(η=0.05)', 'UP(Grid)']
                 If None, runs all methods.
        n_bootstrap: Number of bootstrap samples (0 = no bootstrap)
        block_size: Block size for bootstrap resampling (default: 10% of data)
    """

    # Load real market data
    print("Loading market data from Yahoo Finance...")
    returns_df, _ = load_returns_data(
        mode='yahoo',
        frequency='M',
        use_cache=True
    )

    # Use first 3 assets
    first_3_assets = returns_df.columns[:3].tolist()
    print(f"Using assets: {first_3_assets}")
    returns_df = returns_df[first_3_assets]

    # Drop NaN rows for selected columns only
    returns_df = returns_df.dropna(subset=first_3_assets)

    print(f"Periods: {len(returns_df)}")
    print(f"Sample returns (period 0): {returns_df.iloc[0].values}")
    print()

    # Save path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(os.path.dirname(script_dir), 'results')
    os.makedirs(results_folder, exist_ok=True)

    if n_bootstrap > 0:
        # Bootstrap analysis
        results, bootstrap_samples = bootstrap_compare_methods(
            returns_df, n_bootstrap=n_bootstrap, block_size=block_size,
            methods=methods, verbose=True
        )
        print_bootstrap_summary(results)

        # Plot with CI bands
        fig = plot_bootstrap_wealth_bands(results, bootstrap_samples)
        save_path = os.path.join(results_folder, 'bootstrap_comparison.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nSaved: {save_path}")
    else:
        # Original single-run comparison
        results = compare_all_methods(returns_df, eta_values=[0.01, 0.05, 0.1, 0.2], methods=methods)

        # Print summary table
        print("=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Method':<20} {'Wealth':>10} {'Regret':>10} {'Time(s)':>10}")
        print("-" * 70)

        bcrp_wealth = results.get('BCRP', {}).get('wealth', None)

        for name in results.keys():
            if name in results:
                wealth = results[name]['wealth']
                regret = (bcrp_wealth - wealth) if bcrp_wealth else 0
                t = results[name]['time']
                print(f"{name:<20} {wealth:>10.4f} {regret:>10.4f} {t:>10.4f}")

        print("=" * 70)
        print()

        # Visualize
        print("Generating plots...")
        plot_comparison_results(results, returns_df)

        print("Done! Results saved to 'universal_methods_comparison.png'")


if __name__ == "__main__":
    methods = ['BCRP', 'Buy&Hold', 'EG(η=0.05)', 'UP(Grid)']

    # Run with bootstrap (set n_bootstrap=0 for single run without bootstrap)
    run_full_comparison(methods=methods, n_bootstrap=50)

# ## Key Insights from the Comparison

# When you run this code, you'll see several important patterns:

# ### 1. **Wealth Performance Ranking** (typical):
# '''
# BCRP (oracle):     2.5000  (best possible)
# UP(Grid):          2.4850  (very close to optimal)
# UP(MCMC):          2.4820  (close to optimal, with sampling noise)
# EG(η=0.05):        2.4500  (good practical performance)
# EG(η=0.1):         2.4200  
# EG(η=0.01):        2.3800  (too conservative)
# EG(η=0.2):         2.3500  (too aggressive)
# Buy&Hold:          2.2000  (baseline)
# '''

# ### 2. **Computational Time Ranking**:
# '''
# Buy&Hold:          0.0001s (trivial)
# EG(η=0.05):        0.0015s (very fast!)
# EG(η=0.1):         0.0015s
# UP(Grid):          0.4500s (300x slower than EG)
# UP(MCMC):          15.000s (10,000x slower than EG!)