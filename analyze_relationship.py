import pandas as pd
from scipy.stats import pearsonr, linregress


summary = pd.read_csv("Results/grid_summary.csv")

summary["hhi_advantage"] = ( summary["sample_hhi"] - summary["lw_hhi"]) / summary["sample_hhi"]
summary["realized_vol_advantage"] = ( summary["sample_realized_vol"] - summary["lw_realized_vol"]) / summary["sample_realized_vol"]
summary["forecast_error_advantage"] = ( summary["sample_forecast_error"] - summary["lw_forecast_error"]) / summary["sample_forecast_error"]
summary["turnover_advantage"] = (summary["sample_turnover"] - summary["lw_turnover"]) / summary["sample_turnover"]


metrics = [
    "hhi_advantage",
    "realized_vol_advantage",
    "forecast_error_advantage",
    "turnover_advantage",
]


results = []


for metric in metrics:

    x = summary["N_over_T"]
    y = summary[metric]

    corr, corr_p = pearsonr(x, y)

    regression = linregress(x, y)

    results.append({
        "metric": metric,
        "correlation": corr,
        "correlation_p_value": corr_p,
        "slope_beta": regression.slope,
        "intercept_alpha": regression.intercept,
        "regression_p_value": regression.pvalue,
        "r_squared": regression.rvalue ** 2,
    })


results_df = pd.DataFrame(results)


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(results_df.to_string(index=False))

results_df.to_csv( "results/nt_relationship_results.csv",index=False)
