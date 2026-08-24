from config import START_DATE, END_DATE, RANDOM_SEED

from data import (
    load_nifty500_symbols,
    download_prices,
    filter_by_coverage,
    save_data,
    load_data,
    prices_to_returns,
    select_master_universe,
)

from covariance import (sample_covariance, 
                        condition_number,
                        ledoit_wolf_covariance,
                        )
from portfolio import (gmv_weights, 
                       weight_concentration, 
                       portfolio_returns,
                       predicted_volatility,
                       sharpe_ratio,
                       turnover
)
import numpy as np
import pandas as pd

from experiment import run_walk_forward, run_experiment_grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data onbording

# Loading Nifty 500 symbols
"""symbols = load_nifty500_symbols(
    "data\\ind_nifty500list.csv")
print("Number of symbols:", len(symbols))"""

# Downloading price data for the symbols
"""prices = download_prices(symbols, START_DATE, END_DATE)
save_data(prices, "data\\nifty500_prices.csv")"""

# Filtering stocks based on coverage
'''eligible_prices = filter_by_coverage(prices, min_coverage=1)
save_data(eligible_prices, "data\\eligible_nifty500_prices.csv")

print("Eligible stocks:", eligible_prices.shape[1])'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data pre-proccesing

'''prices = load_data("data\\eligible_nifty500_prices.csv")
returns = prices_to_returns(prices)
print("Returns:", returns.shape)

save_data(returns,"data/returns.csv")

print("\nMissing return fraction:")
print(returns.isna().mean().describe())'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""returns = load_data("data/returns.csv")

master_returns = select_master_universe(
    returns,
    n_assets=150,
    random_seed=RANDOM_SEED
)

save_data(master_returns,"data/master_returns_150.csv")

print("Master universe shape:", master_returns.shape)"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

returns = load_data("data/master_returns_150.csv")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Experimental setting 1.0 
N = 25
T = 126

"""
asset_returns = returns.iloc[:, :N]
window = asset_returns.iloc[:T]
cov_sample = sample_covariance(window)
cov_lw, shrinkage = ledoit_wolf_covariance(window)

sample_kappa, sample_min, sample_max = condition_number(cov_sample)
lw_kappa, lw_min, lw_max = condition_number(cov_lw)
"""

"""
print("\nSample covariance")
print("Condition number:", sample_kappa)
print("Min eigenvalue:", sample_min)
print("Max eigenvalue:", sample_max)


print("\nLedoit-Wolf covariance")
print("Condition number:", lw_kappa)
print("Min eigenvalue:", lw_min)
print("Max eigenvalue:", lw_max)

print("\nLW shrinkage intensity:", shrinkage)

"""
# Result: LW shrinkage intensity: ~15%(0.1523) & improved by ~47%

'''w_sample = gmv_weights(cov_sample)
w_lw = gmv_weights(cov_lw)
sample_hhi = weight_concentration(w_sample)
lw_hhi = weight_concentration(w_lw)
'''
"""
print("\nSample GMV")
print(w_sample.describe())
print("Sum:", w_sample.sum())
print("Sample HHI:", sample_hhi)

print("\nLW GMV")
print(w_lw.describe())
print("Sum:", w_lw.sum())
print("LW HHI:", lw_hhi)

print("Equal-weight HHI:",1 / N)

"""

# result:
"""
Sample GMV
count    25.000000
mean      0.040000
std       0.082116
min      -0.044712
25%      -0.024384
50%       0.020634
75%       0.078893
max       0.290927
dtype: float64
Sum: 1.0
Sample HHI: 0.2018310086878118

LW GMV
count    25.000000
mean      0.040000
std       0.063793
min      -0.037463
25%      -0.006688
50%       0.027806
75%       0.067051
max       0.223690
dtype: float64
Sum: 0.9999999999999997
LW HHI: 0.13766950321258073
Equal-weight HHI: 0.04

"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Experimental setting 1.1
'''holding_period = 21

future_returns = asset_returns.iloc[T:T + holding_period]

sample_port_returns = portfolio_returns(future_returns, w_sample)
lw_port_returns = portfolio_returns(future_returns, w_lw)

sample_realized_vol = sample_port_returns.std() * np.sqrt(252)
lw_realized_vol = lw_port_returns.std() * np.sqrt(252)'''

#print("Sample realized volatility:", sample_realized_vol)
#print("LW realized volatility:", lw_realized_vol)

"""
Results:
Sample realized volatility: 0.12142140435491272
LW realized volatility: 0.11809967346720758

"""

# Computing risk forecast error

'''sample_predicted_vol = predicted_volatility(w_sample,cov_sample)
lw_predicted_vol = predicted_volatility(w_lw,cov_lw)

sample_forecast_error = abs(sample_predicted_vol - sample_realized_vol)
lw_forecast_error = abs(lw_predicted_vol - lw_realized_vol)
'''

"""
print("Sample predicted vol:", sample_predicted_vol)
print("Sample realized vol:", sample_realized_vol)
print("Sample forecast error:", sample_forecast_error)

print("-"*20)

print("LW predicted vol:", lw_predicted_vol)
print("LW realized vol:", lw_realized_vol)
print("LW forecast error:", lw_forecast_error)

"""

"""
Results:

Sample predicted vol: 0.11514691156929754
Sample realized vol: 0.12142140435491272
Sample forecast error: 0.006274492785615177
--------------------
LW predicted vol: 0.12199865781287852
LW realized vol: 0.11809967346720758
LW forecast error: 0.0038989843456709444

"""

# Calculating Sharpe

'''sample_sharpe = sharpe_ratio(sample_port_returns)
lw_sharpe = sharpe_ratio(lw_port_returns)
'''
"""
print("Sample Sharpe:", sample_sharpe)
print("LW Sharpe:", lw_sharpe)
"""

"""
Results:
Sample Sharpe: 3.570187800651217
LW Sharpe: 3.7998531874691492
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Experimental setting 1.2

'''start_1 = 0
start_2 = 21

window_1 = asset_returns.iloc[start_1:start_1 + T]
window_2 = asset_returns.iloc[start_2:start_2 + T]

cov_sample_2 = sample_covariance(window_2)
cov_lw_2, _ = ledoit_wolf_covariance(window_2)

w_sample_2 = gmv_weights(cov_sample_2)
w_lw_2 = gmv_weights(cov_lw_2)

sample_turnover = turnover(w_sample,w_sample_2)
lw_turnover = turnover(w_lw,w_lw_2)
'''
"""
print("Sample turnover:", sample_turnover)
print("LW turnover:", lw_turnover)
"""

"""
Results:
Sample turnover: 0.2665137494791514
LW turnover: 0.1755618539998135

"""
# ===================================================================

# walk-forward loop for N=25,T=126 

'''results = run_walk_forward(returns,N=25,T=126, rebalance_frequency=21)
results.to_csv("Results/results_N25_T126.csv",index=False)

print(results.shape)

summary = results[
    [
        "sample_condition",
        "lw_condition",
        "sample_hhi",
        "lw_hhi",
        "sample_realized_vol",
        "lw_realized_vol",
        "sample_forecast_error",
        "lw_forecast_error",
        "sample_sharpe",
        "lw_sharpe",
        "sample_turnover",
        "lw_turnover",
        "lw_shrinkage",
    ]
].mean()

print(summary)

summary.to_csv("Results/summary_N25_T126.csv",header=["mean"])
'''

"""
Results:
(123, 14)
sample_condition         65.893392
lw_condition             22.319895
sample_hhi                0.214795
lw_hhi                    0.120831
sample_realized_vol       0.154275
lw_realized_vol           0.146288
sample_forecast_error     0.046116
lw_forecast_error         0.035627
sample_sharpe             1.465808
lw_sharpe                 1.654254
sample_turnover           0.316599
lw_turnover               0.197058
lw_shrinkage              0.192224
dtype: float64
"""

"""
So the mechanism is behaving coherently:

Shrinkage → better conditioning → less concentrated/stable weights → lower turnover
"""

# ===================================================================

# walk-forward loop for all 16 combinations

"""
from config import ASSET_COUNTS, LOOKBACK_WINDOWS, REBALANCE_FREQUENCY

results = run_experiment_grid(
    returns=returns,
    asset_counts=ASSET_COUNTS,
    lookback_windows=LOOKBACK_WINDOWS,
    rebalance_frequency=REBALANCE_FREQUENCY,
)

print("\nFinal shape:")
print(results.shape)

results.to_csv(
    "Results/all_experiment_results.csv",
    index=False
)
"""

"""
Results:
Final shape:
(1752, 17)
"""

"""results = pd.read_csv("Results/all_experiment_results.csv")

summary = (
    results.groupby(["N", "T", "N_over_T"]).agg({
        "sample_condition": "mean",
        "lw_condition": "mean",
        "sample_hhi": "mean",
        "lw_hhi": "mean",
        "sample_realized_vol": "mean",
        "lw_realized_vol": "mean",
        "sample_forecast_error": "mean",
        "lw_forecast_error": "mean",
        "sample_turnover": "mean",
        "lw_turnover": "mean",
        "lw_shrinkage": "mean",
    }).reset_index()) # Mean of many 21-day sharpes is not the correct overall portfolio Sharpe so will calculate it later

summary.to_csv(
    "Results/grid_summary.csv",
    index=False
)


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(summary.to_string(index=False))"""


"""
RESULT:
N   T  N_over_T  sample_condition  lw_condition  sample_hhi   lw_hhi  sample_realized_vol  lw_realized_vol  sample_forecast_error  lw_forecast_error  sample_turnover  lw_turnover  lw_shrinkage
 25 126  0.198413      6.589339e+01     22.319895    0.214795 0.120831             0.154275         0.146288               0.046116           0.035627         0.316599     0.197058      0.192224
 25 252  0.099206      4.323154e+01     24.857617    0.169584 0.127221             0.147032         0.142602               0.035939           0.033096         0.150066     0.113032      0.109512
 25 504  0.049603      3.209686e+01     24.408666    0.143581 0.125851             0.147496         0.145646               0.038473           0.037952         0.077120     0.067327      0.063403
 25 756  0.033069      2.890645e+01     24.039337    0.135206 0.123972             0.150287         0.149474               0.042349           0.042190         0.053307     0.048859      0.046544
 50 126  0.396825      2.382071e+02     54.660262    0.214527 0.101277             0.144583         0.124017               0.066404           0.036541         0.638171     0.323117      0.178930
 50 252  0.198413      1.103466e+02     58.655271    0.142034 0.105322             0.125813         0.119577               0.034569           0.029132         0.259068     0.190517      0.100049
 50 504  0.099206      7.479496e+01     55.800751    0.117057 0.102916             0.123032         0.120994               0.029178           0.028640         0.129074     0.111478      0.056501
 50 756  0.066138      6.469072e+01     53.556533    0.110784 0.102029             0.124886         0.123896               0.032488           0.032639         0.089746     0.082020      0.040653
100 126  0.793651      5.028453e+03    137.623729    0.504531 0.079479             0.204677         0.113360               0.166634           0.045043         2.449539     0.503524      0.182123
100 252  0.396825      5.227873e+02    167.681126    0.152355 0.088058             0.125045         0.111158               0.056190           0.035089         0.566854     0.334365      0.101382
100 504  0.198413      2.770557e+02    166.789251    0.106946 0.086187             0.115266         0.111628               0.032657           0.028836         0.244794     0.197193      0.055860
100 756  0.132275      2.286653e+02    165.666646    0.098760 0.086353             0.116653         0.114915               0.031319           0.030401         0.163975     0.143905      0.039223
150 126  1.190476     -2.055045e+16    201.845906 6490.591239 0.067106             4.767966         0.108691               6.566278           0.052857       150.278682     0.622168      0.188971
150 252  0.595238      1.849404e+03    297.676835    0.215542 0.082778             0.138839         0.108765               0.087908           0.044403         1.101600     0.474423      0.105865
150 504  0.297619      5.670821e+02    288.206072    0.113098 0.080819             0.112623         0.107143               0.040496           0.032502         0.374312     0.281197      0.058120
150 756  0.198413      4.190682e+02    279.928400    0.097287 0.080294             0.112707         0.110321               0.033617           0.030993         0.241222     0.205334      0.040641

"""
# FOR 150 126  1.190476  -> [ -2.055045e+16,{N/T>1},{HHI ≈ 6490}, {realized volatility ≈ 477%}, {turnover ≈ 150} ] -> SAMPLE COVARIANCE BECOMES NON.INVERTIBLE ( RANK DEFICIENT -> SINGULAR)
"""
N   T  N_over_T  sample_condition  lw_condition  sample_hhi   lw_hhi  sample_realized_vol  lw_realized_vol  sample_forecast_error  lw_forecast_error  sample_turnover  lw_turnover  lw_shrinkage
 25 126  0.198413         65.893392     22.319895    0.214795 0.120831             0.154275         0.146288               0.046116           0.035627         0.316599     0.197058      0.192224
 25 252  0.099206         43.231538     24.857617    0.169584 0.127221             0.147032         0.142602               0.035939           0.033096         0.150066     0.113032      0.109512
 25 504  0.049603         32.096858     24.408666    0.143581 0.125851             0.147496         0.145646               0.038473           0.037952         0.077120     0.067327      0.063403
 25 756  0.033069         28.906448     24.039337    0.135206 0.123972             0.150287         0.149474               0.042349           0.042190         0.053307     0.048859      0.046544
 50 126  0.396825        238.207068     54.660262    0.214527 0.101277             0.144583         0.124017               0.066404           0.036541         0.638171     0.323117      0.178930
 50 252  0.198413        110.346607     58.655271    0.142034 0.105322             0.125813         0.119577               0.034569           0.029132         0.259068     0.190517      0.100049
 50 504  0.099206         74.794962     55.800751    0.117057 0.102916             0.123032         0.120994               0.029178           0.028640         0.129074     0.111478      0.056501
 50 756  0.066138         64.690717     53.556533    0.110784 0.102029             0.124886         0.123896               0.032488           0.032639         0.089746     0.082020      0.040653
100 126  0.793651       5028.452630    137.623729    0.504531 0.079479             0.204677         0.113360               0.166634           0.045043         2.449539     0.503524      0.182123
100 252  0.396825        522.787269    167.681126    0.152355 0.088058             0.125045         0.111158               0.056190           0.035089         0.566854     0.334365      0.101382
100 504  0.198413        277.055669    166.789251    0.106946 0.086187             0.115266         0.111628               0.032657           0.028836         0.244794     0.197193      0.055860
100 756  0.132275        228.665322    165.666646    0.098760 0.086353             0.116653         0.114915               0.031319           0.030401         0.163975     0.143905      0.039223
150 126  1.190476               inf    201.845906 6490.591239 0.067106             4.767966         0.108691               6.566278           0.052857       150.278682     0.622168      0.188971
150 252  0.595238       1849.404312    297.676835    0.215542 0.082778             0.138839         0.108765               0.087908           0.044403         1.101600     0.474423      0.105865
150 504  0.297619        567.082065    288.206072    0.113098 0.080819             0.112623         0.107143               0.040496           0.032502         0.374312     0.281197      0.058120
150 756  0.198413        419.068194    279.928400    0.097287 0.080294             0.112707         0.110321               0.033617           0.030993         0.241222     0.205334      0.040641
"""

# Using Moore–Penrose pseudoinverse for instead of the ordinary inverse
"""
Results:
Final shape:
(1752, 17)
  N   T  N_over_T  sample_condition  lw_condition  sample_hhi   lw_hhi  sample_realized_vol  lw_realized_vol  sample_forecast_error  lw_forecast_error  sample_turnover  lw_turnover  lw_shrinkage
 25 126  0.198413         65.893392     22.319895    0.214795 0.120831             0.154275         0.146288               0.046116           0.035627         0.316599     0.197058      0.192224
 25 252  0.099206         43.231538     24.857617    0.169584 0.127221             0.147032         0.142602               0.035939           0.033096         0.150066     0.113032      0.109512
 25 504  0.049603         32.096858     24.408666    0.143581 0.125851             0.147496         0.145646               0.038473           0.037952         0.077120     0.067327      0.063403
 25 756  0.033069         28.906448     24.039337    0.135206 0.123972             0.150287         0.149474               0.042349           0.042190         0.053307     0.048859      0.046544
 50 126  0.396825        238.207068     54.660262    0.214527 0.101277             0.144583         0.124017               0.066404           0.036541         0.638171     0.323117      0.178930
 50 252  0.198413        110.346607     58.655271    0.142034 0.105322             0.125813         0.119577               0.034569           0.029132         0.259068     0.190517      0.100049
 50 504  0.099206         74.794962     55.800751    0.117057 0.102916             0.123032         0.120994               0.029178           0.028640         0.129074     0.111478      0.056501
 50 756  0.066138         64.690717     53.556533    0.110784 0.102029             0.124886         0.123896               0.032488           0.032639         0.089746     0.082020      0.040653
100 126  0.793651       5028.452630    137.623729    0.504531 0.079479             0.204677         0.113360               0.166634           0.045043         2.449539     0.503524      0.182123
100 252  0.396825        522.787269    167.681126    0.152355 0.088058             0.125045         0.111158               0.056190           0.035089         0.566854     0.334365      0.101382
100 504  0.198413        277.055669    166.789251    0.106946 0.086187             0.115266         0.111628               0.032657           0.028836         0.244794     0.197193      0.055860
100 756  0.132275        228.665322    165.666646    0.098760 0.086353             0.116653         0.114915               0.031319           0.030401         0.163975     0.143905      0.039223
150 126  1.190476               inf    201.845906 1184.627441 0.067106             4.168377         0.108691               4.920176           0.052857       104.918405     0.622168      0.188971
150 252  0.595238       1849.404312    297.676835    0.215542 0.082778             0.138839         0.108765               0.087908           0.044403         1.101600     0.474423      0.105865
150 504  0.297619        567.082065    288.206072    0.113098 0.080819             0.112623         0.107143               0.040496           0.032502         0.374312     0.281197      0.058120
150 756  0.198413        419.068194    279.928400    0.097287 0.080294             0.112707         0.110321               0.033617           0.030993         0.241222     0.205334      0.040641
"""

# =================================================================================================================


# testing the hypothesis

#     -  LW advantage=(Sample - lw)/ sample
#     -  plot_results.py
#     -  Results are in figures subfolder
#     -  There is "inf" in sample_condition, so will see it seperately


# quantifing the relation b/w N/T & LW Advantage -> analyze_relationship.py

#     -  LW advantage=α+β(N/T)+ϵ  (Simple regression)
#     -  ρ=Corr(N/T,LW advantage) ( pearson correlation)

# Hypothesis is: β > 0

# Results 1 :

"""
                  metric  correlation  correlation_p_value  slope_beta  intercept_alpha  regression_p_value  r_squared
           hhi_advantage     0.958888         4.773804e-09    0.847624         0.084361        4.773804e-09   0.919466
  realized_vol_advantage     0.951228         1.546558e-08    0.758335        -0.098053        1.546558e-08   0.904835
forecast_error_advantage     0.982896         1.097404e-11    0.908414        -0.033453        1.097404e-11   0.966086
      turnover_advantage     0.968217         8.076341e-10    0.819625         0.077864        8.076341e-10   0.937445

"""

"""
Interpretaion:
1. R2=0.966 -> N/T explain 96.6% of variation relative forcast error improvement similarly for others

2. All four relationships are strongly positive:
    -   HHI advantage: r=0.959, R2=0.919
    -   Realized-vol advantage: r=0.951, R2=0.905
    -   Forecast-error advantage: r=0.983, R2=0.966
    -   Turnover advantage: r=0.968, R2=0.937

"""

# For Condition number -> analyze_conditioning.py

# Result 2
"""
SAMPLE COVARIANCE
Slope: 6.3391851764737694
R-squared: 0.8356075941588631
p-value: 1.8763412401340119e-06

LEDOIT-WOLF
Slope: 2.3942663818903616
R-squared: 0.282215009976189
p-value: 0.04156876339419387

LW CONDITIONING ADVANTAGE
Correlation: 0.9210306729917377
Correlation p-value: 1.1059935978972252e-06
Slope: 1.0846034524311488
R-squared: 0.8482975005916128
Regression p-value: 1.105993597897243e-06

SINGULAR SAMPLE COVARIANCE CASES
  N   T  N_over_T  sample_condition  lw_condition
150 126  1.190476               inf    201.845906

"""

"""
Interpretaion:
1. For Ledoit-Wolf, conditioning still worsens as dimensionality increases, but much more slowly
    -   slope sample​ = 6.34 vs LW = 2.39
    -   r=0.921,R2=0.848 
    -   extreme case: κ sample ​= ∞ while for k (LW) ~ 201.85
"""

# Sharpe study -> analyze_sharpe.py

"""
Result:
 N   T  N_over_T  sample_sharpe  lw_sharpe  sharpe_difference
 25 756  0.033069       1.268375   1.276343           0.007968
 25 504  0.049603       1.235363   1.272268           0.036905
 50 756  0.066138       1.664513   1.671115           0.006601
 25 252  0.099206       1.308319   1.374277           0.065958
 50 504  0.099206       1.501559   1.554152           0.052592
100 756  0.132275       1.306200   1.330620           0.024419
 50 252  0.198413       1.412835   1.517881           0.105045
 25 126  0.198413       1.111009   1.162677           0.051668
150 756  0.198413       1.479842   1.530795           0.050952
100 504  0.198413       1.318141   1.391940           0.073799
150 504  0.297619       1.303921   1.469590           0.165669
 50 126  0.396825       1.209865   1.400441           0.190576
100 252  0.396825       1.297900   1.454262           0.156362
150 252  0.595238       1.192019   1.451947           0.259929
100 126  0.793651       0.811988   1.300909           0.488921
150 126  1.190476      -0.594483   1.303379           1.897862

Sharpe advantage correlation: 0.8908669240268107
Correlation p-value: 3.6881840671942713e-06
Slope: 1.3103228744940305
R-squared: 0.7936438763249907
Regression p-value: 3.6881840671943234e-06

"""

"""
Interpretaion:
1. N/T ↑ ⇒ LW Sharpe advantage ↑ (r = 0.89, R2= 0.79)
    -   N>T -> the sample-covariance GMV breaks down, while LW still produces a reasonable portfolio
    -   79% of the variation in Sharpe improvement across the 16 settings is associated with N/T.
    -   p≈3.7×10−6 -> positive trend is statistically very strong within this experimental grid
"""


# Hypothesis conclusion:
#   ->  Across the controlled N/T grid, Ledoit-Wolf’s advantage increased strongly as N/T rose. 
#       Correlations were 0.95–0.98 for concentration, turnover, realized volatility, and risk-forecast error, 
#       0.92 for covariance-conditioning advantage, and 0.89 for full out-of-sample Sharpe improvement.


