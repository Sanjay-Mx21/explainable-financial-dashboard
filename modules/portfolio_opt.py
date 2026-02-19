import numpy as np

def mean_variance_opt(returns, target_return=None):
    X = returns
    try:
        import pandas as pd
        if hasattr(X, "values"):
            mu = X.mean().values
            Sigma = X.cov().values
        else:
            mu = np.mean(X, axis=0)
            Sigma = np.cov(X, rowvar=False)
        n = len(mu)
    except Exception:
        n = X.shape[1] if hasattr(X, "shape") else 1
        return np.ones(n)/n

    try:
        from scipy.optimize import minimize
        def port_var(w): return w.T @ Sigma @ w
        cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
        if target_return is not None:
            cons = cons + ({'type':'eq','fun': lambda w: w.dot(mu) - target_return},)
        bounds = tuple((0,1) for _ in range(len(mu)))
        x0 = np.ones(len(mu))/len(mu)
        res = minimize(port_var, x0, bounds=bounds, constraints=cons)
        if res.success:
            return np.maximum(res.x, 0) / np.sum(np.maximum(res.x, 0))
        else:
            return np.ones(len(mu))/len(mu)
    except Exception:
        pos = np.clip(mu, a_min=0, a_max=None)
        s = pos.sum()
        if s <= 0:
            return np.ones(len(mu))/len(mu)
        return pos / s
