import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time



class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion model with characteristic function and moment extraction
    """
    
    def __init__(self, S0, mu, sigma, lam, mu_J, sigma_J):
        """
        Parameters:
        S0: Initial stock price
        mu: Drift rate
        sigma: Volatility of diffusion
        lam: Jump intensity (jumps per unit time)
        mu_J: Mean of log-jump sizes
        sigma_J: Std of log-jump sizes
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
    
    def characteristic_function(self, u, t):
        """
        Characteristic function of log-price X(t) = ln(S(t))
        φ(u,t) = E[exp(iuX(t))]
        """
        # Exponent components
        initial_term = 1j * u * np.log(self.S0)
        
        drift_term = 1j * u * (self.mu - 0.5 * self.sigma**2) * t
        
        diffusion_term = -0.5 * u**2 * self.sigma**2 * t
        
        jump_cf = np.exp(1j * u * self.mu_J - 0.5 * u**2 * self.sigma_J**2)
        jump_term = self.lam * t * (jump_cf - 1)
        
        exponent = initial_term + drift_term + diffusion_term + jump_term
        
        return np.exp(exponent)
    
    def analytical_moments_log_price(self, t):
        """
        Analytical formulas for moments of log-price X(t)
        """
        # First moment (mean)
        mean = np.log(self.S0) + t * (self.mu - 0.5*self.sigma**2 + self.lam*self.mu_J)
        
        # Second central moment (variance)
        variance = t * (self.sigma**2 + self.lam * (self.mu_J**2 + self.sigma_J**2))
        
        # Third central moment (for skewness)
        third_moment = t * self.lam * (self.mu_J**3 + 3*self.mu_J*self.sigma_J**2)
        
        # Fourth central moment (for kurtosis)
        fourth_moment = t * self.lam * (self.mu_J**4 + 6*self.mu_J**2*self.sigma_J**2 + 3*self.sigma_J**4)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'skewness': third_moment / variance**(3/2) if variance > 0 else 0,
            'excess_kurtosis': fourth_moment / variance**2 - 3 if variance > 0 else 0
        }
    
    def numerical_derivative(self, func, x, n=1, h=1e-5):
        """
        Compute numerical derivative using finite differences
        More robust than scipy.misc.derivative (which is deprecated)
        
        Uses central difference for better accuracy:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        """
        if n == 1:
            # First derivative - central difference
            return (func(x + h) - func(x - h)) / (2 * h)
        elif n == 2:
            # Second derivative
            return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h)
        elif n == 3:
            # Third derivative - using 5-point stencil
            return (func(x + 2*h) - 2*func(x + h) + 2*func(x - h) - func(x - 2*h)) / (2 * h**3)
        elif n == 4:
            # Fourth derivative - using 5-point stencil  
            return (func(x + 2*h) - 4*func(x + h) + 6*func(x) - 4*func(x - h) + func(x - 2*h)) / (h**4)
        else:
            raise ValueError(f"Derivative order {n} not implemented")
    
    def numerical_moments_from_cf(self, t, max_moment=4):
        """
        Extract moments numerically from characteristic function using derivatives
        Updated to use custom numerical differentiation
        """
        def cf_func(u):
            return self.characteristic_function(u, t)
        
        moments = {}
        
        # Calculate raw moments using (-i)^n * d^n φ(u)/du^n |_{u=0}
        for n in range(1, max_moment + 1):
            try:
                # Split into real and imaginary parts for more stable differentiation
                def cf_real(u):
                    return cf_func(u).real
                
                def cf_imag(u):
                    return cf_func(u).imag
                
                # Compute derivatives of real and imaginary parts separately
                real_deriv = self.numerical_derivative(cf_real, 0, n=n)
                imag_deriv = self.numerical_derivative(cf_imag, 0, n=n)
                
                # Combine using (-i)^n * (real_deriv + i * imag_deriv)
                derivative_at_zero = real_deriv + 1j * imag_deriv
                raw_moment = ((-1j)**n * derivative_at_zero).real
                
                moments[f'raw_moment_{n}'] = raw_moment
                
            except Exception as e:
                print(f"Warning: Error calculating moment {n}: {e}")
                moments[f'raw_moment_{n}'] = np.nan
        
        # Convert to central moments
        if 'raw_moment_1' in moments and not np.isnan(moments['raw_moment_1']):
            mean = moments['raw_moment_1']
            moments['mean'] = mean
            
            if 'raw_moment_2' in moments and not np.isnan(moments['raw_moment_2']):
                variance = moments['raw_moment_2'] - mean**2
                moments['variance'] = max(0, variance)  # Ensure non-negative
                moments['std'] = np.sqrt(moments['variance'])
                
                if ('raw_moment_3' in moments and not np.isnan(moments['raw_moment_3']) 
                    and moments['variance'] > 1e-10):
                    third_central = (moments['raw_moment_3'] - 3*mean*moments['raw_moment_2'] 
                                   + 2*mean**3)
                    moments['skewness'] = third_central / (moments['variance']**(3/2))
                    
                    if ('raw_moment_4' in moments and not np.isnan(moments['raw_moment_4'])):
                        fourth_central = (moments['raw_moment_4'] - 4*mean*moments['raw_moment_3']
                                        + 6*mean**2*moments['raw_moment_2'] - 3*mean**4)
                        if moments['variance'] > 1e-10:
                            moments['excess_kurtosis'] = fourth_central / (moments['variance']**2) - 3
        
        return moments
    
    def price_moments(self, t):
        """
        Analytical moments for price S(t) = exp(X(t))
        """
        # Mean price: E[S(t)] = φ(-i, t)
        mean_price = self.characteristic_function(-1j, t).real
        
        # Second moment: E[S(t)^2] = φ(-2i, t)
        second_moment_price = self.characteristic_function(-2j, t).real
        
        variance_price = second_moment_price - mean_price**2
        
        return {
            'mean_price': mean_price,
            'variance_price': variance_price,
            'std_price': np.sqrt(variance_price),
            'cv_price': np.sqrt(variance_price) / mean_price  # coefficient of variation
        }
    
    def simulate_paths(self, t, n_paths=100000, n_steps=252):
        """
        Monte Carlo simulation of the Merton jump-diffusion process
        
        Parameters:
        t: Time horizon
        n_paths: Number of simulation paths
        n_steps: Number of time steps per unit time (e.g., 252 for daily steps in a year)
        
        Returns:
        log_prices: Array of log-prices X(t) at time t
        prices: Array of prices S(t) at time t
        """
        dt = t / n_steps
        total_steps = int(n_steps * t)
        
        # Initialize arrays
        log_prices = np.full(n_paths, np.log(self.S0))
        
        # Precompute constants
        drift_dt = (self.mu - 0.5 * self.sigma**2) * dt
        vol_sqrt_dt = self.sigma * np.sqrt(dt)
        
        # Generate random numbers in batches for efficiency
        np.random.seed(42)  # For reproducibility
        
        # Brownian increments
        dW = np.random.normal(0, 1, (n_paths, total_steps)) * vol_sqrt_dt
        
        # Jump times and sizes
        for step in range(total_steps):
            # Add diffusion component
            log_prices += drift_dt + dW[:, step]
            
            # Add jumps (Poisson arrivals)
            jump_indicator = np.random.poisson(self.lam * dt, n_paths)
            has_jumps = jump_indicator > 0
            
            if np.any(has_jumps):
                # For paths with jumps, generate jump sizes
                n_jumps_total = np.sum(jump_indicator)
                if n_jumps_total > 0:
                    jump_sizes = np.random.normal(self.mu_J, self.sigma_J, n_jumps_total)
                    
                    # Assign jumps to paths
                    jump_idx = 0
                    for i in range(n_paths):
                        if jump_indicator[i] > 0:
                            # Sum all jumps that occur in this time step for this path
                            path_jumps = jump_sizes[jump_idx:jump_idx + jump_indicator[i]]
                            log_prices[i] += np.sum(path_jumps)
                            jump_idx += jump_indicator[i]
        
        prices = np.exp(log_prices)
        
        return log_prices, prices
    
    def monte_carlo_moments(self, t, n_paths=100000, n_steps=252):
        """
        Calculate moments using Monte Carlo simulation
        """
        print(f"Running Monte Carlo with {n_paths:,} paths...")
        start_time = time.time()
        
        log_prices, prices = self.simulate_paths(t, n_paths, n_steps)
        
        simulation_time = time.time() - start_time
        print(f"Simulation completed in {simulation_time:.2f} seconds")
        
        # Calculate moments for log-prices
        log_moments = {
            'mean': np.mean(log_prices),
            'variance': np.var(log_prices, ddof=1),
            'std': np.std(log_prices, ddof=1),
            'skewness': stats.skew(log_prices),
            'excess_kurtosis': stats.kurtosis(log_prices)
            # 'excess_kurtosis': stats.kurtosis(log_prices, excess=True)
        }
        
        # Calculate moments for prices
        price_moments = {
            'mean_price': np.mean(prices),
            'variance_price': np.var(prices, ddof=1),
            'std_price': np.std(prices, ddof=1),
            'cv_price': np.std(prices, ddof=1) / np.mean(prices)
        }
        
        # Add confidence intervals (95%)
        n = len(log_prices)
        se_mean = log_moments['std'] / np.sqrt(n)
        se_var = log_moments['std'] * np.sqrt(2/(n-1))  # Approximate
        
        log_moments['mean_ci'] = (log_moments['mean'] - 1.96*se_mean, 
                                 log_moments['mean'] + 1.96*se_mean)
        log_moments['variance_ci'] = (max(0, log_moments['variance'] - 1.96*se_var), 
                                     log_moments['variance'] + 1.96*se_var)
        
        return log_moments, price_moments, {'log_prices': log_prices, 'prices': prices}













def compare_all_methods(model, t, n_paths=100000):
    """
    Compare analytical, numerical CF, and Monte Carlo moment extraction
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MOMENT COMPARISON at t = {t}")
    print(f"{'='*60}")
    
    # 1. Analytical moments
    print("\n1. ANALYTICAL MOMENTS (Log-Price):")
    analytical = model.analytical_moments_log_price(t)
    for key, value in analytical.items():
        print(f"   {key:15}: {value:10.6f}")
    
    # 2. Numerical moments from CF
    print("\n2. NUMERICAL MOMENTS FROM CF (Log-Price):")
    numerical = model.numerical_moments_from_cf(t)
    for key, value in numerical.items():
        if not np.isnan(value):
            print(f"   {key:15}: {value:10.6f}")
    
    # 3. Monte Carlo moments
    print(f"\n3. MONTE CARLO SIMULATION ({n_paths:,} paths):")
    mc_log, mc_price, mc_data = model.monte_carlo_moments(t, n_paths)
    print("   Log-Price Moments:")
    for key, value in mc_log.items():
        if key.endswith('_ci'):
            print(f"   {key:15}: ({value[0]:.6f}, {value[1]:.6f})")
        else:
            print(f"   {key:15}: {value:10.6f}")
    
    # 4. Price moments comparison
    print("\n4. PRICE PROCESS MOMENTS:")
    analytical_price = model.price_moments(t)
    print("   Method          Mean Price    Std Price     CV")
    print("   ------------  ------------  -----------  ------")
    print(f"   Analytical    {analytical_price['mean_price']:12.4f}  {analytical_price['std_price']:11.4f}  {analytical_price['cv_price']:6.4f}")
    print(f"   Monte Carlo   {mc_price['mean_price']:12.4f}  {mc_price['std_price']:11.4f}  {mc_price['cv_price']:6.4f}")
    
    # 5. Detailed comparison table
    print(f"\n5. DETAILED COMPARISON (Log-Price):")
    print("   Metric         Analytical    CF Numerical   Monte Carlo    MC 95% CI          Error (MC-Anal)")
    print("   -----------  ------------  -------------  ------------  -----------------  ----------------")
    
    metrics = ['mean', 'variance', 'skewness', 'excess_kurtosis']
    for metric in metrics:
        if metric in analytical and metric in mc_log:
            anal_val = analytical[metric]
            mc_val = mc_log[metric]
            mc_ci = mc_log.get(f'{metric}_ci', ('N/A', 'N/A'))
            num_val = numerical.get(metric, np.nan)
            error = mc_val - anal_val
            
            ci_str = f"({mc_ci[0]:.6f}, {mc_ci[1]:.6f})" if mc_ci != ('N/A', 'N/A') else "N/A"
            num_str = f"{num_val:10.6f}" if not np.isnan(num_val) else "    N/A   "
            
            print(f"   {metric:11}  {anal_val:12.6f}  {num_str}  {mc_val:12.6f}  {ci_str:17}  {error:10.6f}")
    
    # 6. Statistical tests
    print(f"\n6. VALIDATION TESTS:")
    
    # Test if MC mean is within confidence interval of analytical
    mc_mean = mc_log['mean']
    mc_mean_ci = mc_log['mean_ci']
    anal_mean = analytical['mean']
    
    within_ci = mc_mean_ci[0] <= anal_mean <= mc_mean_ci[1]
    print(f"   Analytical mean within MC 95% CI: {within_ci}")
    
    # Test if MC variance is close to analytical
    mc_var = mc_log['variance']
    mc_var_ci = mc_log['variance_ci']
    anal_var = analytical['variance']
    
    within_var_ci = mc_var_ci[0] <= anal_var <= mc_var_ci[1]
    print(f"   Analytical variance within MC 95% CI: {within_var_ci}")
    
    # Relative errors
    mean_rel_error = abs(mc_mean - anal_mean) / abs(anal_mean) * 100
    var_rel_error = abs(mc_var - anal_var) / abs(anal_var) * 100
    
    print(f"   Relative error in mean: {mean_rel_error:.4f}%")
    print(f"   Relative error in variance: {var_rel_error:.4f}%")
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'monte_carlo': mc_log,
        'price_analytical': analytical_price,
        'price_monte_carlo': mc_price,
        'simulation_data': mc_data
    }

def plot_distributions_comparison(model, t, results):
    """
    Plot histograms of simulated data vs theoretical density (interactive, non-blocking)
    """
    mc_data = results['simulation_data']
    log_prices = mc_data['log_prices']
    prices = mc_data['prices']
    
    # Create interactive figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Jump-Diffusion Analysis at t={t}', fontsize=16, fontweight='bold')
    
    # Enable interactive navigation
    fig.canvas.toolbar_visible = True
    
    # 1. Log-price histogram vs theoretical moments
    ax1 = axes[0, 0]
    n_bins = 50
    counts, bins, patches = ax1.hist(log_prices, bins=n_bins, density=True, alpha=0.7, 
                                   color='skyblue', label='MC Simulation', edgecolor='black', linewidth=0.5)
    
    # Overlay normal approximation using theoretical moments
    anal_moments = results['analytical']
    x_range = np.linspace(log_prices.min(), log_prices.max(), 1000)
    normal_approx = stats.norm.pdf(x_range, anal_moments['mean'], anal_moments['std'])
    line1 = ax1.plot(x_range, normal_approx, 'r-', linewidth=3, label='Normal Approximation')
    
    vline1 = ax1.axvline(anal_moments['mean'], color='red', linestyle='--', alpha=0.8, 
                        linewidth=2, label='Theoretical Mean')
    vline2 = ax1.axvline(results['monte_carlo']['mean'], color='blue', linestyle='--', 
                        alpha=0.8, linewidth=2, label='MC Mean')
    
    ax1.set_xlabel('Log Price X(t)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Log-Price Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean: {results["monte_carlo"]["mean"]:.4f}\nStd: {results["monte_carlo"]["std"]:.4f}\nSkew: {results["monte_carlo"]["skewness"]:.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Price histogram (interactive)
    ax2 = axes[0, 1]
    counts2, bins2, patches2 = ax2.hist(prices, bins=n_bins, density=True, alpha=0.7, 
                                       color='lightgreen', label='MC Simulation', 
                                       edgecolor='black', linewidth=0.5)
    
    # Color bars based on value for visual appeal
    for i, (patch, count) in enumerate(zip(patches2, counts2)):
        patch.set_facecolor(plt.cm.viridis(count / max(counts2)))
    
    # Theoretical mean lines
    vline3 = ax2.axvline(results['price_analytical']['mean_price'], color='red', 
                        linestyle='--', alpha=0.8, linewidth=2, label='Theoretical Mean')
    vline4 = ax2.axvline(results['price_monte_carlo']['mean_price'], color='blue', 
                        linestyle='--', alpha=0.8, linewidth=2, label='MC Mean')
    
    ax2.set_xlabel('Price S(t)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Price Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add price statistics
    price_stats = f'Mean: ${results["price_monte_carlo"]["mean_price"]:.2f}\nStd: ${results["price_monte_carlo"]["std_price"]:.2f}\nCV: {results["price_monte_carlo"]["cv_price"]:.4f}'
    ax2.text(0.02, 0.98, price_stats, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Interactive Q-Q plot for log-prices vs normal
    ax3 = axes[1, 0]
    stats.probplot(log_prices, dist="norm", 
                   sparams=(anal_moments['mean'], anal_moments['std']), 
                   plot=ax3)
    ax3.set_title('Q-Q Plot: Log-Prices vs Normal', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(log_prices)), 
                                         anal_moments['mean'], anal_moments['std'])
    sorted_log_prices = np.sort(log_prices)
    correlation = np.corrcoef(sorted_log_prices, theoretical_quantiles)[0, 1]
    ax3.text(0.02, 0.98, f'R² = {correlation**2:.4f}', transform=ax3.transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 4. Interactive sample path evolution with zoom capability
    ax4 = axes[1, 1]
    
    # Generate sample paths for visualization
    n_sample_paths = 8
    n_steps = 252
    dt = t / n_steps
    time_grid = np.linspace(0, t, n_steps + 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_sample_paths))
    
    for i in range(n_sample_paths):
        np.random.seed(100 + i)  # Different seed for each path
        log_path = np.full(n_steps + 1, np.log(model.S0))
        
        for step in range(n_steps):
            # Brownian increment
            dW = np.random.normal(0, np.sqrt(dt)) * model.sigma
            
            # Jump component
            n_jumps = np.random.poisson(model.lam * dt)
            jump_total = 0
            if n_jumps > 0:
                jumps = np.random.normal(model.mu_J, model.sigma_J, n_jumps)
                jump_total = np.sum(jumps)
            
            # Update log price
            log_path[step + 1] = (log_path[step] + 
                                 (model.mu - 0.5*model.sigma**2)*dt + 
                                 dW + jump_total)
        
        price_path = np.exp(log_path)
        ax4.plot(time_grid, price_path, alpha=0.8, linewidth=2, 
                color=colors[i], label=f'Path {i+1}' if i < 3 else "")
    
    # Reference lines
    ax4.axhline(model.S0, color='black', linestyle='-', alpha=0.7, linewidth=2, 
               label='Initial Price')
    ax4.axhline(results['price_analytical']['mean_price'], color='red', 
               linestyle='--', alpha=0.9, linewidth=3, label='Theoretical Mean')
    
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('Price', fontsize=12)
    ax4.set_title('Sample Price Paths', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Add interactivity - zoom rectangle selector
    from matplotlib.widgets import RectangleSelector
    
    def onselect(eclick, erelease):
        """Callback for rectangle selector"""
        ax4.set_xlim(eclick.xdata, erelease.xdata)
        ax4.set_ylim(eclick.ydata, erelease.ydata)
        fig.canvas.draw()
    
    # Rectangle selector for zooming (comment out if causing issues)
    try:
        selector = RectangleSelector(ax4, onselect, useblit=True, button=[1], 
                                   minspanx=5, minspany=5, spancoords='pixels', 
                                   interactive=True)
    except:
        pass  # Skip if widget causes issues
    
    plt.tight_layout()
    
    # Make it non-blocking and interactive
    plt.draw()
    plt.pause(0.001)  # Small pause to ensure rendering
    
    # Add instructions
    print(f"📊 Interactive plots created for t={t}")
    print("   • Use mouse to pan and zoom")
    print("   • Close plots when ready to continue")
    print("   • Rectangle selector available on sample paths plot")
    
    return fig

def convergence_test(model, t, path_sizes=[1000, 5000, 10000, 50000, 100000]):
    """
    Test Monte Carlo convergence as number of paths increases (interactive plots)
    """
    print(f"\n{'='*50}")
    print(f"MONTE CARLO CONVERGENCE TEST at t = {t}")
    print(f"{'='*50}")

    analytical = model.analytical_moments_log_price(t)
    theoretical_mean = analytical['mean']
    theoretical_var = analytical['variance']
    
    print(f"Theoretical Mean: {theoretical_mean:.6f}")
    print(f"Theoretical Variance: {theoretical_var:.6f}")
    print()
    print("N Paths      MC Mean     Error(Mean)    MC Variance   Error(Var)    Time(s)")
    print("-" * 75)
    
    convergence_data = []
    
    for n_paths in path_sizes:
        start_time = time.time()
        mc_log, _, _ = model.monte_carlo_moments(t, n_paths, n_steps=100)  # Fewer steps for speed
        elapsed = time.time() - start_time
        
        mean_error = abs(mc_log['mean'] - theoretical_mean)
        var_error = abs(mc_log['variance'] - theoretical_var)
        
        print(f"{n_paths:7,}    {mc_log['mean']:9.6f}   {mean_error:10.6f}   {mc_log['variance']:10.6f}   {var_error:9.6f}   {elapsed:7.2f}")
        
        convergence_data.append({
            'n_paths': n_paths,
            'mc_mean': mc_log['mean'],
            'mc_var': mc_log['variance'],
            'mean_error': mean_error,
            'var_error': var_error,
            'time': elapsed
        })
    
    # Create interactive convergence plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Monte Carlo Convergence Analysis (t={t})', fontsize=16, fontweight='bold')
    
    n_paths_arr = [d['n_paths'] for d in convergence_data]
    mean_errors = [d['mean_error'] for d in convergence_data]
    var_errors = [d['var_error'] for d in convergence_data]
    times = [d['time'] for d in convergence_data]
    
    # Mean convergence with interactive features
    ax1 = axes[0]
    line1 = ax1.loglog(n_paths_arr, mean_errors, 'bo-', markersize=8, linewidth=2, 
                      label='Mean Error', markerfacecolor='lightblue', markeredgecolor='blue')
    line2 = ax1.loglog(n_paths_arr, [1/np.sqrt(n) * mean_errors[0] * np.sqrt(n_paths_arr[0]) 
                                    for n in n_paths_arr], 'r--', linewidth=2, 
                      label='1/√n theoretical', alpha=0.8)
    
    ax1.set_xlabel('Number of Paths', fontsize=12)
    ax1.set_ylabel('Absolute Error in Mean', fontsize=12)
    ax1.set_title('Mean Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4, which='both')
    
    # Add data cursor for interactivity
    for i, (x, y) in enumerate(zip(n_paths_arr, mean_errors)):
        ax1.annotate(f'{y:.2e}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
    
    # Variance convergence with enhanced visuals
    ax2 = axes[1]
    line3 = ax2.loglog(n_paths_arr, var_errors, 'go-', markersize=8, linewidth=2, 
                      label='Variance Error', markerfacecolor='lightgreen', markeredgecolor='green')
    
    ax2.set_xlabel('Number of Paths', fontsize=12)
    ax2.set_ylabel('Absolute Error in Variance', fontsize=12)
    ax2.set_title('Variance Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4, which='both')
    
    # Add trend line
    if len(var_errors) > 1:
        # Fit power law: error ∝ n^(-α)
        log_n = np.log(n_paths_arr)
        log_err = np.log(var_errors)
        slope, intercept = np.polyfit(log_n, log_err, 1)
        trend_line = np.exp(intercept) * np.array(n_paths_arr)**slope
        ax2.loglog(n_paths_arr, trend_line, 'm:', linewidth=2, alpha=0.7,
                  label=f'Fitted: n^{slope:.2f}')
        ax2.legend(fontsize=10)
    
    # Computational time with interactive elements
    ax3 = axes[2]
    bars = ax3.bar(range(len(n_paths_arr)), times, color=plt.cm.plasma(np.linspace(0, 1, len(times))),
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Path Size Index', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(n_paths_arr)))
    ax3.set_xticklabels([f'{n:,}' for n in n_paths_arr], rotation=45)
    ax3.grid(True, alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    
    print(f"\n📈 Interactive convergence plots created")
    print("   • Hover over points for detailed values")
    print("   • Use toolbar to zoom and pan")
    
    return convergence_data






    #######################
    # Example usage
if __name__ == "__main__":
    # Model parameters
    S0 = 100        # Initial price
    mu = 0.05       # Drift (5% per year)
    sigma = 0.2     # Volatility (20% per year)
    lam = 0.1       # Jump intensity (0.1 jumps per year on average)
    mu_J = -0.1     # Average log-jump size (negative = downward bias)
    sigma_J = 0.15  # Jump size volatility
    
    # Create model
    model = MertonJumpDiffusion(S0, mu, sigma, lam, mu_J, sigma_J)
    
    print("MERTON JUMP-DIFFUSION MODEL")
    print("=" * 40)
    print(f"Parameters:")
    print(f"  S0 = {S0}, μ = {mu}, σ = {sigma}")
    print(f"  λ = {lam}, μ_J = {mu_J}, σ_J = {sigma_J}")
    
    # Main comparison at 1 year
    t = 1.0
    print(f"\n🚀 Starting comprehensive analysis...")
    results = compare_all_methods(model, t, n_paths=100000)
    
    # Create interactive visualizations
    print(f"\n📊 Creating interactive visualizations...")
    fig1 = plot_distributions_comparison(model, t, results)
    
    # Convergence test with interactive plots
    print(f"\n🔄 Running convergence analysis...")
    convergence_data = convergence_test(model, t)
    
    # Test different parameter scenarios
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT SCENARIOS")
    print(f"{'='*60}")
    
    scenarios = [
        {"name": "High Jump Frequency", "lam": 0.5, "mu_J": -0.05, "sigma_J": 0.1},
        {"name": "Large Negative Jumps", "lam": 0.1, "mu_J": -0.3, "sigma_J": 0.2},
        {"name": "No Jumps (Pure GBM)", "lam": 0.0, "mu_J": 0, "sigma_J": 0},
    ]
    
    scenario_figs = []
    for i, scenario in enumerate(scenarios):
        print(f"\n--- {scenario['name']} ---")
        test_model = MertonJumpDiffusion(S0, mu, sigma, 
                                       scenario['lam'], scenario['mu_J'], scenario['sigma_J'])
        
        # Quick comparison with fewer paths for speed
        test_results = compare_all_methods(test_model, t, n_paths=50000)
        
        # Create comparison plot for this scenario
        fig_scenario = plot_distributions_comparison(test_model, t, test_results)
        fig_scenario.suptitle(f'{scenario["name"]} - Jump-Diffusion Analysis', 
                             fontsize=16, fontweight='bold')
        scenario_figs.append(fig_scenario)
    
    print(f"\n{'='*60}")
    print("🎯 INTERACTIVE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Key features of interactive plots:")
    print("✓ Non-blocking execution - continue working while plots are open")
    print("✓ Pan and zoom with mouse/toolbar")
    print("✓ Rectangle selector on sample paths for detailed zooming")
    print("✓ Color-coded histograms and enhanced visual elements")
    print("✓ Hover annotations on convergence plots")
    print("✓ Statistical information boxes")
    print("")
    print("📋 Summary insights:")
    print("• Characteristic function provides exact analytical moments")
    print("• Monte Carlo converges to theoretical values (~1/√n rate)")
    print("• Jump parameters significantly affect higher-order moments")
    print("• Interactive plots help visualize jump-diffusion behavior")
    print("")
    print("🔧 Next steps you could explore:")
    print("• Modify parameters in real-time and re-run")
    print("• Implement option pricing using the characteristic function")
    print("• Add parameter estimation from real market data")
    print("• Extend to more complex jump models (stochastic volatility)")
    
    # Keep plots open and provide user control
    print(f"\n💡 Tip: Close individual plot windows when done, or close all to exit")
    print("      The program will continue running until you close plots or interrupt")
    
    # Interactive menu for continued exploration
    while True:
        try:
            print(f"\n" + "="*50)
            print("INTERACTIVE EXPLORATION MENU")
            print("="*50)
            print("1. Test custom parameters")
            print("2. Quick convergence test")
            print("3. Generate new sample paths")
            print("4. Compare time horizons")
            print("5. Exit and close all plots")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\nEnter custom parameters:")
                try:
                    custom_lam = float(input(f"Jump intensity λ (current: {lam}): ") or lam)
                    custom_mu_J = float(input(f"Jump mean μ_J (current: {mu_J}): ") or mu_J)
                    custom_sigma_J = float(input(f"Jump volatility σ_J (current: {sigma_J}): ") or sigma_J)
                    
                    custom_model = MertonJumpDiffusion(S0, mu, sigma, custom_lam, custom_mu_J, custom_sigma_J)
                    print(f"\n🔄 Running analysis with custom parameters...")
                    custom_results = compare_all_methods(custom_model, t, n_paths=75000)
                    custom_fig = plot_distributions_comparison(custom_model, t, custom_results)
                    custom_fig.suptitle(f'Custom Parameters: λ={custom_lam}, μ_J={custom_mu_J}, σ_J={custom_sigma_J}', 
                                       fontsize=14, fontweight='bold')
                except ValueError:
                    print("❌ Invalid input. Using default parameters.")
                    
            elif choice == '2':
                print("\n🔄 Running quick convergence test...")
                quick_convergence = convergence_test(model, t, path_sizes=[5000, 20000, 50000])
                
            elif choice == '3':
                print("\n🎲 Generating new sample paths...")
                new_results = compare_all_methods(model, t, n_paths=50000)
                new_fig = plot_distributions_comparison(model, t, new_results)
                new_fig.suptitle('New Sample Paths - Jump-Diffusion Analysis', 
                               fontsize=16, fontweight='bold')
                
            elif choice == '4':
                print("\nComparing different time horizons...")
                time_horizons = [0.25, 0.5, 1.0, 2.0]
                
                fig_time, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig_time.suptitle('Time Horizon Comparison', fontsize=16, fontweight='bold')
                
                for i, t_compare in enumerate(time_horizons):
                    ax = axes[i//2, i%2]
                    
                    # Quick simulation for each time horizon
                    log_prices, prices = model.simulate_paths(t_compare, n_paths=30000, n_steps=100)
                    
                    ax.hist(log_prices, bins=30, density=True, alpha=0.7, 
                           color=plt.cm.viridis(i/len(time_horizons)), label=f't = {t_compare}')
                    
                    # Theoretical mean and std
                    anal_moments = model.analytical_moments_log_price(t_compare)
                    ax.axvline(anal_moments['mean'], color='red', linestyle='--', alpha=0.8)
                    
                    ax.set_title(f'Log-Price Distribution at t = {t_compare}')
                    ax.set_xlabel('Log Price')
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)
                
            elif choice == '5':
                print("\n👋 Closing all plots and exiting...")
                plt.close('all')
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully...")
            plt.close('all')
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            continue
    
    print("\n✅ Analysis complete. Thank you for exploring jump-diffusion models!")


# Configure matplotlib for interactive plotting
plt.ion()  # Turn on interactive mode
plt.style.use('default')  # Use default style for better interactivity