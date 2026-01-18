import torch
import numpy as np
import pandas as pd
from ..data.synthetic import generate_regime_switching_returns
from ..inference.denoise import denoise_portfolio

class SyntheticPortfolioEvaluator:
    """Evaluate portfolio performance on synthetic data."""
    
    def evaluate_portfolio_performance(self, model, test_returns, lookback=60, 
                                      rebalance_freq=5, device='cpu'):
        """
        Backtest portfolio performance.
        
        Returns:
            Dict with final_wealth, sharpe, max_drawdown, portfolios
        """
        n_periods = len(test_returns) - lookback
        
        wealth = 1.0
        wealths = [wealth]
        portfolios = []
        
        for t in range(0, n_periods, rebalance_freq):
            # Get allocation
            returns_history = torch.tensor(
                test_returns[t:t+lookback], 
                dtype=torch.float32
            )
            b_t, _ = denoise_portfolio(model, returns_history, device=device)
            portfolios.append(b_t)
            
            # Hold for rebalance_freq periods
            for tau in range(rebalance_freq):
                if t + lookback + tau >= len(test_returns):
                    break
                period_return = test_returns[t + lookback + tau]
                wealth *= (b_t @ period_return)
                wealths.append(wealth)
        
        # Metrics
        wealths = np.array(wealths)
        returns = wealths[1:] / wealths[:-1] - 1
        
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        max_dd = (wealths / np.maximum.accumulate(wealths) - 1).min()
        
        return {
            'final_wealth': wealth,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'portfolios': portfolios
        }
    
    def compare_to_baselines(self, model, test_returns, device='cpu'):
        """Compare to equal-weight baseline."""
        n_assets = test_returns.shape[1]
        
        # Model performance
        model_perf = self.evaluate_portfolio_performance(
            model, test_returns, device=device
        )
        
        # Equal weight baseline
        b_equal = np.ones(n_assets) / n_assets
        wealth_equal = 1.0
        for ret in test_returns[60:]:
            wealth_equal *= (b_equal @ ret)
        returns_equal = test_returns[60:] @ b_equal
        sharpe_equal = returns_equal.mean() / (returns_equal.std() + 1e-8) * np.sqrt(252)
        
        results = {
            'hybrid_denoiser': model_perf,
            'equal_weight': {
                'final_wealth': wealth_equal,
                'sharpe': sharpe_equal,
                'max_drawdown': np.nan
            }
        }
        
        return pd.DataFrame(results).T

def quick_evaluation_suite(model, n_runs=5, device='cpu'):
    """Run quick evaluation on synthetic data."""
    evaluator = SyntheticPortfolioEvaluator()
    results = []
    
    print("Running quick evaluation suite...")
    
    for run in range(n_runs):
        test_returns, _ = generate_regime_switching_returns(
            n_periods=1000, 
            seed=run
        )
        
        perf = evaluator.evaluate_portfolio_performance(
            model, test_returns, device=device
        )
        
        results.append({
            'run': run,
            'final_wealth': perf['final_wealth'],
            'sharpe': perf['sharpe'],
            'max_drawdown': perf['max_drawdown']
        })
        
        print(f"Run {run+1}/{n_runs}: Sharpe={perf['sharpe']:.3f}, "
              f"Wealth={perf['final_wealth']:.3f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n=== Evaluation Summary ===")
    print(f"Mean Sharpe: {results_df['sharpe'].mean():.3f} ± {results_df['sharpe'].std():.3f}")
    print(f"Mean Final Wealth: {results_df['final_wealth'].mean():.3f}")
    
    return results_df


