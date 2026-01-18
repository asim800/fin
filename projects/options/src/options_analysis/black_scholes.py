"""Black-Scholes option pricing and implied volatility calculation module."""

import logging
import math
from typing import Tuple, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar


class BlackScholesCalculator:
    """Black-Scholes option pricing model implementation."""
    
    def __init__(self):
        """Initialize BlackScholesCalculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_option_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.
        
        This replicates the R BSMcall function.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if T <= 0:
            # Handle expired options
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            self.logger.warning("Volatility must be positive")
            return 0.0
        
        try:
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                # Call option price
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                # Put option price
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0.0)  # Price cannot be negative
            
        except Exception as e:
            self.logger.error(f"Error calculating BS price: {e}")
            return 0.0
    
    def calculate_both_prices(self, S: float, K: float, T: float, r: float, 
                            sigma: float) -> Tuple[float, float]:
        """
        Calculate both call and put prices (replicates R BSMcall function exactly).
        
        Args:
            S: Current stock price
            K: Strike price  
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (call_price, put_price)
        """
        call_price = self.calculate_option_price(S, K, T, r, sigma, 'call')
        put_price = self.calculate_option_price(S, K, T, r, sigma, 'put')
        
        return call_price, put_price
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str = 'call',
                                   max_iterations: int = 1000, tolerance: float = 0.01) -> float:
        """
        Calculate implied volatility using iterative method.
        
        This replicates the R findVol function.
        
        Args:
            market_price: Market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        if market_price <= 0:
            return 0.0
        
        if T <= 0:
            return 0.0
        
        # Initial guess
        vol = 0.1
        alpha = 0.01  # Learning rate
        
        for i in range(max_iterations):
            try:
                # Calculate theoretical price
                theoretical_price = self.calculate_option_price(S, K, T, r, vol, option_type)
                
                # Calculate error
                error = theoretical_price - market_price
                
                # Check convergence
                if abs(error) <= tolerance:
                    return vol
                
                # Update volatility using gradient descent approach
                vol = vol - alpha * error
                
                # Keep volatility positive
                if vol <= 0:
                    vol = 0.001
                
                # Prevent volatility from becoming too large
                if vol > 5.0:
                    vol = 5.0
                
            except Exception as e:
                self.logger.warning(f"Error in IV calculation iteration {i}: {e}")
                break
        
        self.logger.warning(f"IV calculation did not converge after {max_iterations} iterations")
        return vol
    
    def calculate_implied_volatility_scipy(self, market_price: float, S: float, K: float,
                                         T: float, r: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using scipy optimization.
        
        Args:
            market_price: Market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        if market_price <= 0 or T <= 0:
            return 0.0
        
        def objective(vol):
            """Objective function to minimize."""
            try:
                theoretical_price = self.calculate_option_price(S, K, T, r, vol, option_type)
                return abs(theoretical_price - market_price)
            except:
                return float('inf')
        
        try:
            # Use scipy to find optimal volatility
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
            
            if result.success:
                return result.x
            else:
                # Fallback to iterative method
                return self.calculate_implied_volatility(market_price, S, K, T, r, option_type)
                
        except Exception as e:
            self.logger.error(f"Error in scipy IV calculation: {e}")
            # Fallback to iterative method
            return self.calculate_implied_volatility(market_price, S, K, T, r, option_type)
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> dict:
        """
        Calculate option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with Greeks
        """
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Common terms
            sqrt_T = math.sqrt(T)
            exp_minus_rT = math.exp(-r * T)
            phi_d1 = norm.pdf(d1)
            Phi_d1 = norm.cdf(d1)
            Phi_d2 = norm.cdf(d2)
            
            if option_type.lower() == 'call':
                # Call Greeks
                delta = Phi_d1
                theta = (-S * phi_d1 * sigma / (2 * sqrt_T) - 
                        r * K * exp_minus_rT * Phi_d2) / 365  # Per day
                rho = K * T * exp_minus_rT * Phi_d2 / 100  # Per 1% change
            else:
                # Put Greeks
                delta = Phi_d1 - 1
                theta = (-S * phi_d1 * sigma / (2 * sqrt_T) + 
                        r * K * exp_minus_rT * norm.cdf(-d2)) / 365  # Per day
                rho = -K * T * exp_minus_rT * norm.cdf(-d2) / 100  # Per 1% change
            
            # Gamma and Vega are same for calls and puts
            gamma = phi_d1 / (S * sigma * sqrt_T)
            vega = S * phi_d1 * sqrt_T / 100  # Per 1% change in volatility
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def calculate_time_to_expiry(self, expiry_date_str: str) -> float:
        """
        Calculate time to expiry in years from expiry date string.
        
        Args:
            expiry_date_str: Expiry date in format "MMM.DD.YYYY" (e.g., "Feb.21.2025")
            
        Returns:
            Time to expiry in years
        """
        try:
            from datetime import datetime
            
            # Parse the date format from R (e.g., "Feb.21.2025")
            expiry_date = datetime.strptime(expiry_date_str, "%b.%d.%Y")
            current_date = datetime.now()
            
            # Calculate difference in days
            days_to_expiry = (expiry_date - current_date).days
            
            # Convert to years (using 365 days per year)
            time_to_expiry = max(days_to_expiry / 365.0, 0)
            
            return time_to_expiry
            
        except Exception as e:
            self.logger.error(f"Error calculating time to expiry for {expiry_date_str}: {e}")
            return 0.0
    
    def build_density_analysis(self, returns: list, threshold: float, ndays: int = 1) -> dict:
        """
        Build density analysis for returns below threshold.
        
        This replicates the R buildDensity function.
        
        Args:
            returns: List of returns
            threshold: Threshold for filtering returns
            ndays: Number of days to look ahead
            
        Returns:
            Dictionary with analysis results
        """
        if not returns:
            return {'filtered_returns': [], 'original_threshold_returns': []}
        
        # Find indices where returns are below threshold
        below_threshold_indices = [i for i, ret in enumerate(returns) if ret < threshold]
        
        if not below_threshold_indices:
            return {'filtered_returns': [], 'original_threshold_returns': []}
        
        # Get next day(s) returns after threshold violations
        next_day_returns = []
        
        for idx in below_threshold_indices:
            for day_offset in range(1, ndays + 1):
                next_idx = idx + day_offset
                if next_idx < len(returns):
                    next_day_returns.append(returns[next_idx])
        
        original_threshold_returns = [returns[i] for i in below_threshold_indices]
        
        return {
            'filtered_returns': next_day_returns,
            'original_threshold_returns': original_threshold_returns,
            'threshold': threshold,
            'ndays': ndays,
            'num_violations': len(below_threshold_indices)
        }