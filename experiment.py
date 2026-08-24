import numpy as np
import pandas as pd

from covariance import (
    sample_covariance,
    ledoit_wolf_covariance,
    condition_number,
)

from portfolio import (
    gmv_weights,
    weight_concentration,
    portfolio_returns,
    predicted_volatility,
    sharpe_ratio,
    turnover,
)


def run_walk_forward(returns, N, T, rebalance_frequency=21):

    asset_returns = returns.iloc[:, :N]

    results = []

    previous_sample_weights = None
    previous_lw_weights = None

    for start in range(0,len(asset_returns) - T - rebalance_frequency + 1,rebalance_frequency,):

        # Estimation window
        train = asset_returns.iloc[start:start + T]
        # Future holding window
        future = asset_returns.iloc[start + T:start + T + rebalance_frequency]

        # ===================================================================

        # SAMPLE COVARIANCE
        cov_sample = sample_covariance(train)
        sample_condition, _, _ = condition_number(cov_sample)
        w_sample = gmv_weights(cov_sample)
        sample_hhi = weight_concentration(w_sample)
        sample_future_returns = portfolio_returns(future,w_sample)
        sample_realized_vol = (sample_future_returns.std()* np.sqrt(252))
        sample_predicted_vol = predicted_volatility(w_sample,cov_sample)
        sample_forecast_error = abs(sample_predicted_vol- sample_realized_vol)
        sample_sharpe = sharpe_ratio( sample_future_returns)

        if previous_sample_weights is None:
            sample_turnover = np.nan
        else:
            sample_turnover = turnover(previous_sample_weights,w_sample)

        # ===================================================================

        # LEDOIT-WOLF
        cov_lw, shrinkage = ledoit_wolf_covariance(train)
        lw_condition, _, _ = condition_number(cov_lw)
        w_lw = gmv_weights( cov_lw)
        lw_hhi = weight_concentration( w_lw)
        lw_future_returns = portfolio_returns(future,w_lw)
        lw_realized_vol = (lw_future_returns.std()* np.sqrt(252))
        lw_predicted_vol = predicted_volatility(w_lw,cov_lw)
        lw_forecast_error = abs(lw_predicted_vol- lw_realized_vol)
        lw_sharpe = sharpe_ratio(lw_future_returns)

        if previous_lw_weights is None:
            lw_turnover = np.nan
        else:
            lw_turnover = turnover(previous_lw_weights,w_lw)

        # ===================================================================

        # Save this rebalance result

        results.append({
            "date": future.index[0],

            "sample_condition": sample_condition,
            "lw_condition": lw_condition,

            "sample_hhi": sample_hhi,
            "lw_hhi": lw_hhi,

            "sample_realized_vol": sample_realized_vol,
            "lw_realized_vol": lw_realized_vol,

            "sample_forecast_error": sample_forecast_error,
            "lw_forecast_error": lw_forecast_error,

            "sample_sharpe": sample_sharpe,
            "lw_sharpe": lw_sharpe,

            "sample_turnover": sample_turnover,
            "lw_turnover": lw_turnover,

            "lw_shrinkage": shrinkage,
        })

        previous_sample_weights = w_sample
        previous_lw_weights = w_lw

    return pd.DataFrame(results)

def run_experiment_grid(
    returns,
    asset_counts,
    lookback_windows,
    rebalance_frequency=21,
):
    all_results = []

    for N in asset_counts:
        for T in lookback_windows:

            print(f"Running N={N}, T={T}")

            results = run_walk_forward(returns=returns,N=N,T=T,rebalance_frequency=rebalance_frequency)

            results["N"] = N
            results["T"] = T
            results["N_over_T"] = N / T

            all_results.append(results)

    return pd.concat(
        all_results,
        ignore_index=True
    )

        # ===================================================================

# For full out-of-sample Sharpe

def run_walk_forward_with_returns(returns, N, T, rebalance_frequency=21):
    asset_returns = returns.iloc[:, :N]

    sample_oos_returns = []
    lw_oos_returns = []

    for start in range(0, len(asset_returns) - T - rebalance_frequency + 1, rebalance_frequency):
        train = asset_returns.iloc[start:start + T]
        future = asset_returns.iloc[start + T:start + T + rebalance_frequency]

        cov_sample = sample_covariance(train)
        w_sample = gmv_weights(cov_sample)
        sample_returns = portfolio_returns(future, w_sample)

        cov_lw, _ = ledoit_wolf_covariance(train)
        w_lw = gmv_weights(cov_lw)
        lw_returns = portfolio_returns(future, w_lw)

        sample_oos_returns.append(sample_returns)
        lw_oos_returns.append(lw_returns)

    sample_oos_returns = pd.concat(sample_oos_returns)
    lw_oos_returns = pd.concat(lw_oos_returns)

    return sample_oos_returns, lw_oos_returns