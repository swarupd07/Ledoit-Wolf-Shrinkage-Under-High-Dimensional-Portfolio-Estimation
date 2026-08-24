import os
import numpy as np
import pandas as pd

from data import load_data
from experiment import run_walk_forward_with_returns
from config import ASSET_COUNTS, LOOKBACK_WINDOWS, REBALANCE_FREQUENCY
from scipy.stats import pearsonr, linregress


returns = load_data("data/master_returns_150.csv")
results = []

os.makedirs("Results/oos_returns", exist_ok=True)

for N in ASSET_COUNTS:
    for T in LOOKBACK_WINDOWS:
        print(f"Running Sharpe: N={N}, T={T}")

        sample_returns, lw_returns = run_walk_forward_with_returns(
            returns=returns,
            N=N,
            T=T,
            rebalance_frequency=REBALANCE_FREQUENCY,
        )

        sample_sharpe = (sample_returns.mean() / sample_returns.std()) * np.sqrt(252)
        lw_sharpe = (lw_returns.mean() / lw_returns.std()) * np.sqrt(252)

        oos = pd.DataFrame({
            "sample_return": sample_returns,
            "lw_return": lw_returns,
        })

        oos.to_csv(f"Results/oos_returns/oos_N{N}_T{T}.csv")

        results.append({
            "N": N,
            "T": T,
            "N_over_T": N / T,
            "sample_sharpe": sample_sharpe,
            "lw_sharpe": lw_sharpe,
            "sharpe_difference": lw_sharpe - sample_sharpe,
        })

sharpe_results = pd.DataFrame(results)

sharpe_results.to_csv("Results/full_oos_sharpe.csv", index=False)

pd.set_option("display.max_columns", None)

print("\nFull OOS Sharpe Results\n")
print(sharpe_results.sort_values("N_over_T").to_string(index=False))

# Quantifying sharpe

x = sharpe_results["N_over_T"]
y = sharpe_results["sharpe_difference"]

corr, corr_p = pearsonr(x, y)
reg = linregress(x, y)

print("Sharpe advantage correlation:", corr)
print("Correlation p-value:", corr_p)
print("Slope:", reg.slope)
print("R-squared:", reg.rvalue ** 2)
print("Regression p-value:", reg.pvalue)