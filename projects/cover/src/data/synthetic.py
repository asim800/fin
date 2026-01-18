import numpy as np

def generate_regime_switching_returns(n_periods=1000, n_assets=5, seed=42):
    """
    Generate synthetic returns with known regime switches.
    
    Returns:
        returns: Price relatives [n_periods, n_assets]
        regimes: List of regime labels
    """
    np.random.seed(seed)
    
    # Define regimes
    regime_params = {
        'bull': {
            'mean': np.array([0.001, 0.0008, 0.0005, 0.0003, 0.0002]),
            'vol': 0.01
        },
        'bear': {
            'mean': np.array([-0.001, -0.0005, 0.0, 0.0002, 0.0003]),
            'vol': 0.02
        },
        'sideways': {
            'mean': np.zeros(n_assets),
            'vol': 0.015
        }
    }
    
    # Create regime sequence
    regime_length = n_periods // 5
    regime_sequence = (
        ['bull'] * regime_length +
        ['bear'] * regime_length +
        ['sideways'] * regime_length +
        ['bull'] * regime_length +
        ['bear'] * (n_periods - 4 * regime_length)
    )
    
    returns = []
    regimes = []
    
    for regime in regime_sequence:
        params = regime_params[regime]
        ret = np.random.normal(params['mean'], params['vol'])
        returns.append(1 + ret)  # Price relatives
        regimes.append(regime)
    
    return np.array(returns), regimes


