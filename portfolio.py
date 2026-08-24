import numpy as np
import pandas as pd

def gmv_weights(cov_matrix):
    sigma = cov_matrix.values

    n = sigma.shape[0]
    ones = np.ones(n)

    # inv_sigma = np.linalg.inv(sigma)    # create instablility when covar is not rull rank (N/T > 1)
    inv_sigma = np.linalg.pinv(sigma)

    numerator = inv_sigma @ ones
    denominator = ones @ inv_sigma @ ones

    weights = numerator / denominator

    return pd.Series( weights,index=cov_matrix.index)

def weight_concentration(weights):
    return (weights ** 2).sum()

def portfolio_returns(returns, weights):
    return returns @ weights

def predicted_volatility(weights, cov_matrix, annualization=252):
    w = weights.values
    sigma = cov_matrix.values

    variance = w @ sigma @ w

    return np.sqrt(variance * annualization)


def sharpe_ratio(portfolio_returns, annualization=252):
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    if volatility == 0:
        return np.nan

    return (mean_return / volatility) * np.sqrt(annualization)

def turnover(old_weights, new_weights):
    return 0.5 * (new_weights - old_weights).abs().sum()