import os
import pandas as pd
import matplotlib.pyplot as plt


summary = pd.read_csv("Results/grid_summary.csv")


summary["condition_advantage"] = ( summary["sample_condition"] - summary["lw_condition"]) / summary["sample_condition"]
summary["hhi_advantage"] = ( summary["sample_hhi"] - summary["lw_hhi"]) / summary["sample_hhi"]
summary["realized_vol_advantage"] = ( summary["sample_realized_vol"] - summary["lw_realized_vol"]) / summary["sample_realized_vol"]
summary["forecast_error_advantage"] = ( summary["sample_forecast_error"] - summary["lw_forecast_error"]) / summary["sample_forecast_error"]
summary["turnover_advantage"] = (summary["sample_turnover"] - summary["lw_turnover"]) / summary["sample_turnover"]


os.makedirs("figures", exist_ok=True)

# Plot function

def plot_advantage(column, ylabel, filename):

    plot_data = summary.sort_values("N_over_T")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        plot_data["N_over_T"],
        plot_data[column]
    )

    plt.plot(
        plot_data["N_over_T"],
        plot_data[column],
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("N / T")
    plt.ylabel(ylabel)

    plt.title(
        f"Ledoit-Wolf Advantage vs N/T\n{ylabel}"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/{filename}",
        dpi=300
    )

    plt.close()


# Generating plots

plot_advantage(
    "hhi_advantage",
    "Relative Reduction in Weight Concentration",
    "hhi_advantage_vs_nt.png"
)

plot_advantage(
    "realized_vol_advantage",
    "Relative Reduction in Realized Volatility",
    "realized_vol_advantage_vs_nt.png"
)

plot_advantage(
    "forecast_error_advantage",
    "Relative Reduction in Risk Forecast Error",
    "forecast_error_advantage_vs_nt.png"
)

plot_advantage(
    "turnover_advantage",
    "Relative Reduction in Turnover",
    "turnover_advantage_vs_nt.png"
)


print("Saved all plots to figures/")