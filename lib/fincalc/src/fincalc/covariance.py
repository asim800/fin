"""
Covariance matrix estimation methods.

Provides multiple covariance estimation approaches:
- Sample covariance
- Exponentially weighted
- Ledoit-Wolf shrinkage
- Robust (MCD)
- Factor model
- Sparse inverse (graphical lasso)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from sklearn.covariance import LedoitWolf, MinCovDet

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class CovarianceEstimator:
    """
    Covariance matrix estimator with multiple methods.

    Supports sample, exponentially weighted, shrunk, robust,
    factor model, and sparse inverse covariance estimation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._last_precision_matrix = None

    def estimate(self, returns: pd.DataFrame, method: str = 'sample',
                 **kwargs) -> pd.DataFrame:
        """
        Estimate covariance matrix using specified method.

        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        method : str
            Covariance method: 'sample', 'exponential_weighted',
            'shrunk', 'robust', 'factor_model', 'sparse_inverse'
        **kwargs : dict
            Method-specific parameters

        Returns:
        --------
        pd.DataFrame: Covariance matrix with asset names
        """
        method_map = {
            'sample': self._sample,
            'exponential_weighted': self._exponential_weighted,
            'shrunk': self._shrunk,
            'robust': self._robust,
            'factor_model': self._factor_model,
            'sparse_inverse': self._sparse_inverse,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method '{method}'. Available: {list(method_map.keys())}")

        cov_matrix = method_map[method](returns, **kwargs)

        # Ensure positive semi-definite
        cov_matrix = self._ensure_psd(cov_matrix)

        # Convert to DataFrame
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

    def _sample(self, returns: pd.DataFrame) -> np.ndarray:
        """Sample covariance matrix."""
        cov = returns.cov().values
        self.logger.info("Calculated sample covariance matrix")
        return cov

    def _exponential_weighted(self, returns: pd.DataFrame,
                              alpha: float = 0.94) -> np.ndarray:
        """Exponentially weighted covariance matrix."""
        ewm_cov = returns.ewm(alpha=alpha).cov().iloc[-len(returns.columns):].values
        self.logger.info(f"Calculated exponentially weighted covariance (alpha={alpha})")
        return ewm_cov

    def _shrunk(self, returns: pd.DataFrame,
                shrinkage: Optional[float] = None) -> np.ndarray:
        """Ledoit-Wolf shrunk covariance matrix."""
        lw = LedoitWolf(shrinkage=shrinkage)
        cov_matrix = lw.fit(returns.values).covariance_
        self.logger.info(f"Calculated Ledoit-Wolf shrunk covariance (shrinkage={lw.shrinkage_:.3f})")
        return cov_matrix

    def _robust(self, returns: pd.DataFrame, method: str = 'mcd') -> np.ndarray:
        """Robust covariance matrix (Minimum Covariance Determinant)."""
        if method == 'mcd':
            robust_cov = MinCovDet().fit(returns.values)
            cov_matrix = robust_cov.covariance_
            self.logger.info("Calculated robust covariance using MCD")
        else:
            raise ValueError(f"Unknown robust method: {method}")
        return cov_matrix

    def _factor_model(self, returns: pd.DataFrame,
                      factors: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Factor model covariance matrix."""
        if factors is None:
            market_factor = returns.mean(axis=1)
            factors = pd.DataFrame({'Market': market_factor})

        n_assets = len(returns.columns)
        factor_loadings = np.zeros((n_assets, len(factors.columns)))
        residual_vars = np.zeros(n_assets)

        for i, asset in enumerate(returns.columns):
            y = returns[asset].values
            X = np.column_stack([np.ones(len(factors)), factors.values])

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                factor_loadings[i] = beta[1:]
                predicted = X @ beta
                residuals = y - predicted
                residual_vars[i] = np.var(residuals)
            except np.linalg.LinAlgError:
                factor_loadings[i] = 0
                residual_vars[i] = returns[asset].var()

        factor_cov = factors.cov().values
        systematic_cov = factor_loadings @ factor_cov @ factor_loadings.T
        idiosyncratic_cov = np.diag(residual_vars)

        cov_matrix = systematic_cov + idiosyncratic_cov
        self.logger.info(f"Calculated factor model covariance using {len(factors.columns)} factors")
        return cov_matrix

    def _sparse_inverse(self, returns: pd.DataFrame, alpha: float = 0.1,
                        max_iters: int = 1000) -> np.ndarray:
        """Sparse inverse covariance using graphical lasso."""
        if not CVXPY_AVAILABLE:
            self.logger.warning("cvxpy not available, falling back to sample covariance")
            return self._sample(returns)

        try:
            S = np.cov(returns.T)
            n_assets = S.shape[0]

            Theta = cp.Variable((n_assets, n_assets), symmetric=True)
            L1_mask = np.ones((n_assets, n_assets)) - np.eye(n_assets)

            objective = cp.Maximize(
                cp.log_det(Theta) - cp.trace(S @ Theta) -
                alpha * cp.norm(cp.multiply(L1_mask, Theta), 1)
            )

            constraints = [Theta >> 1e-8 * np.eye(n_assets)]
            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver=cp.SCS, max_iters=max_iters, verbose=False)
            except cp.SolverError:
                problem.solve(solver=cp.ECOS, max_iters=max_iters, verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                self.logger.warning(f"Optimization failed: {problem.status}")
                return self._sample(returns)

            precision_matrix = Theta.value
            if precision_matrix is None:
                return self._sample(returns)

            precision_matrix = (precision_matrix + precision_matrix.T) / 2
            cov_matrix = np.linalg.inv(precision_matrix)

            self._last_precision_matrix = precision_matrix
            sparsity = np.sum(np.abs(precision_matrix) < 1e-6) / (n_assets ** 2)
            self.logger.info(f"Calculated sparse inverse covariance (sparsity={sparsity:.1%})")

            return cov_matrix

        except Exception as e:
            self.logger.error(f"Error in sparse inverse: {e}")
            return self._sample(returns)

    def _ensure_psd(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_precision_matrix(self) -> Optional[np.ndarray]:
        """Get the last computed precision matrix (from sparse_inverse)."""
        return self._last_precision_matrix

    @staticmethod
    def available_methods() -> List[str]:
        """Get list of available covariance methods."""
        return ['sample', 'exponential_weighted', 'shrunk', 'robust',
                'factor_model', 'sparse_inverse']

    @staticmethod
    def method_parameters(method: str) -> Dict[str, Any]:
        """Get parameters for a specific method."""
        params = {
            'sample': {},
            'exponential_weighted': {'alpha': 0.94},
            'shrunk': {'shrinkage': None},
            'robust': {'method': 'mcd'},
            'factor_model': {'factors': None},
            'sparse_inverse': {'alpha': 0.1, 'max_iters': 1000},
        }
        return params.get(method, {})
