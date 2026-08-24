import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

def sample_covariance(returns_window):
    return returns_window.cov()   # Pandas implements std. 1/(T-1) sample covar


def condition_number(cov_matrix, tol=1e-12):
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    lambda_min = eigenvalues.min()
    lambda_max = eigenvalues.max()

    if lambda_min <= tol:
        kappa = np.inf
    else:
        kappa = lambda_max / lambda_min

    return kappa, lambda_min, lambda_max


def ledoit_wolf_covariance(returns_window):
    estimator = LedoitWolf()

    estimator.fit(returns_window.values)

    cov_matrix = pd.DataFrame(
        estimator.covariance_,
        index=returns_window.columns,
        columns=returns_window.columns
    )

    return cov_matrix, estimator.shrinkage_