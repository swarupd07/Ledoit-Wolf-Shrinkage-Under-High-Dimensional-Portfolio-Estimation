import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


summary = pd.read_csv("Results/grid_summary.csv")

os.makedirs("figures", exist_ok=True)

# Identifing finite sample cases

finite_sample = np.isfinite(
    summary["sample_condition"]
)

finite_df = summary[finite_sample].copy()

singular_df = summary[~finite_sample].copy()

# Plot condition numbers

plot_df = finite_df.sort_values("N_over_T")

plt.figure(figsize=(8, 5))

plt.scatter(
    plot_df["N_over_T"],
    plot_df["sample_condition"],
    label="Sample covariance"
)

plt.scatter(
    plot_df["N_over_T"],
    plot_df["lw_condition"],
    label="Ledoit-Wolf"
)

plt.yscale("log")

plt.xlabel("N / T")
plt.ylabel("Condition Number (log scale)")

plt.title(
    "Covariance Conditioning vs N/T"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/condition_number_vs_nt.png",
    dpi=300
)

plt.close()



# Relative improvement -> finite cases only

finite_df["condition_advantage"] = (
    finite_df["sample_condition"]
    - finite_df["lw_condition"]
) / finite_df["sample_condition"]



# Regression: N/T vs log(condition number)
x = finite_df["N_over_T"]
sample_log_condition = np.log(
    finite_df["sample_condition"]
)

lw_log_condition = np.log(
    finite_df["lw_condition"]
)


sample_reg = linregress(
    x,
    sample_log_condition
)

lw_reg = linregress(
    x,
    lw_log_condition
)



# Regression: N/T vs LW condition advantage

adv_reg = linregress(
    x,
    finite_df["condition_advantage"]
)

adv_corr, adv_corr_p = pearsonr(
    x,
    finite_df["condition_advantage"]
)


print("\nSAMPLE COVARIANCE")
print("Slope:", sample_reg.slope)
print("R-squared:", sample_reg.rvalue ** 2)
print("p-value:", sample_reg.pvalue)


print("\nLEDOIT-WOLF")
print("Slope:", lw_reg.slope)
print("R-squared:", lw_reg.rvalue ** 2)
print("p-value:", lw_reg.pvalue)


print("\nLW CONDITIONING ADVANTAGE")
print("Correlation:", adv_corr)
print("Correlation p-value:", adv_corr_p)
print("Slope:", adv_reg.slope)
print("R-squared:", adv_reg.rvalue ** 2)
print("Regression p-value:", adv_reg.pvalue)


# -------------------------
# Print singular cases
# -------------------------

print("\nSINGULAR SAMPLE COVARIANCE CASES")

if len(singular_df) == 0:
    print("None")
else:
    print(
        singular_df[
            [
                "N",
                "T",
                "N_over_T",
                "sample_condition",
                "lw_condition",
            ]
        ].to_string(index=False)
    )


# -------------------------
# Save finite results
# -------------------------

finite_df.to_csv(
    "Results/condition_number_analysis.csv",
    index=False
)
