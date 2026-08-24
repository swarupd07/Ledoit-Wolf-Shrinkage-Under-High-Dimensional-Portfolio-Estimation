# Ledoit-Wolf vs Sample Covariance in GMV Portfolios

> **A controlled experiment testing whether Ledoit-Wolf covariance shrinkage becomes more valuable as portfolio dimensionality (N/T) increases.**

![Status](https://img.shields.io/badge/status-complete-brightgreen) ![Universe](https://img.shields.io/badge/universe-NIFTY%20500-blue) ![Configs](https://img.shields.io/badge/configurations-16-orange)

---

## Table of Contents

1. [Research Question](#research-question)
2. [Project Structure](#project-structure)
3. [Pipeline](#pipeline)
   - [1. Data Onboarding](#1-data-onboarding)
   - [2. Data Coverage Filtering](#2-data-coverage-filtering)
   - [3. Convert Prices to Returns](#3-convert-prices-to-returns)
   - [4. Fixed 150-Stock Master Universe](#4-fixed-150-stock-master-universe)
   - [5. Initial Controlled Example: N = 25, T = 126](#5-initial-controlled-example-n--25-t--126)
   - [6. Sample Covariance vs Ledoit-Wolf](#6-sample-covariance-vs-ledoit-wolf)
   - [7. GMV Portfolio Construction](#7-gmv-portfolio-construction)
   - [8. Out-of-Sample Realized Volatility](#8-out-of-sample-realized-volatility)
   - [9. Risk Forecast Error](#9-risk-forecast-error)
   - [10. Single-Window Sharpe](#10-single-window-sharpe)
   - [11. Turnover](#11-turnover)
   - [12. Walk-Forward Experiment for N = 25, T = 126](#12-walk-forward-experiment-for-n--25-t--126)
   - [13. Full 16-Setting Experiment](#13-full-16-setting-experiment)
   - [14. Grid Summary](#14-grid-summary)
   - [15. Singular Sample Covariance at N > T](#15-singular-sample-covariance-at-n--t)
   - [16. Ledoit-Wolf Advantage Metrics](#16-ledoit-wolf-advantage-metrics)
   - [17. Relationship Between N/T and Ledoit-Wolf Advantage](#17-relationship-between-nt-and-ledoit-wolf-advantage)
   - [18. Condition Number Analysis](#18-condition-number-analysis)
   - [19. Full Out-of-Sample Sharpe](#19-full-out-of-sample-sharpe)
4. [Main Empirical Finding](#20-main-empirical-finding)
5. [Mechanism](#21-mechanism)
6. [Final Conclusion](#22-final-conclusion)
7. [Important Limitations](#23-important-limitations)
8. [Main Execution Order](#24-main-execution-order)
9. [Key Output Files](#25-key-output-files)

---

## Research Question

**Does the advantage of Ledoit-Wolf covariance shrinkage increase as the dimensionality ratio $N/T$ increases?**

| Symbol | Meaning |
|:---:|---|
| $N$ | Number of assets in the portfolio |
| $T$ | Number of historical observations used to estimate covariance |

**Controlled grid**

$$N \in \{25,\ 50,\ 100,\ 150\} \qquad T \in \{126,\ 252,\ 504,\ 756\}$$

This gives **16** $(N, T)$ settings spanning a wide range of $N/T$ ratios.

> **Hypothesis:** Ledoit-Wolf advantage increases as $N/T \uparrow$

---

## Project Structure

```text
project/
│
├── data/
│   ├── eligible_nifty500_prices.csv
│   ├── ind_nifty500list.csv
│   ├── master_returns_150.csv
│   ├── nifty500_prices.csv
│   └── returns.csv
│
├── figures/
│   ├── condition_number_vs_nt.png
│   ├── forecast_error_advantage_vs_nt.png
│   ├── hhi_advantage_vs_nt.png
│   ├── realized_vol_advantage_vs_nt.png
│   └── turnover_advantage_vs_nt.png
│
├── Results/
│   ├── oos_returns/
│   ├── all_experiment_results.csv
│   ├── condition_number_analysis.csv
│   ├── full_oos_sharpe.csv
│   ├── grid_summary.csv
│   ├── nt_relationship_results.csv
│   ├── results_N25_T126.csv
│   └── summary_N25_T126.csv
│
├── analyze_conditioning.py
├── analyze_relationship.py
├── analyze_sharpe.py
├── config.py
├── covariance.py
├── data.py
├── experiment1.py
├── main.py
├── plot_results.py
├── portfolio.py
└── requirements.txt
```

---

## Pipeline

### 1. Data Onboarding

The experiment uses the **NIFTY 500** as the starting stock universe.

**Execution**

```python
symbols = load_nifty500_symbols("data\\ind_nifty500list.csv")
print("Number of symbols:", len(symbols))
```

**Result**

```text
Number of symbols: 500
```

Historical prices are downloaded and stored once:

```python
prices = download_prices(symbols, START_DATE, END_DATE)
save_data(prices, "data\\nifty500_prices.csv")
```

Resulting raw price matrix:

```text
Shape: (2715, 500)
```

---

### 2. Data Coverage Filtering

To avoid missing-value complications inside rolling covariance windows, only assets with complete price coverage are retained.

**Execution**

```python
eligible_prices = filter_by_coverage(prices, min_coverage=1)
save_data(eligible_prices, "data\\eligible_nifty500_prices.csv")
print("Eligible stocks:", eligible_prices.shape[1])
```

**Result**

```text
Eligible stocks: 309
```

309 stocks have complete price history across the selected period — comfortably above the maximum experimental requirement of $N = 150$.

---

### 3. Convert Prices to Returns

Covariance is estimated from returns rather than price levels.

For asset $i$:

$$r_{t,i} = \frac{P_{t,i}}{P_{t-1,i}} - 1$$

**Execution**

```python
prices = load_data("data\\eligible_nifty500_prices.csv")
returns = prices_to_returns(prices)
save_data(returns, "data/returns.csv")
```

**Result**

```text
Returns: (2714, 309)

Missing return fraction:
count    309.0
mean       0.0
std        0.0
min        0.0
25%        0.0
50%        0.0
75%        0.0
max        0.0
```

The resulting return matrix has **zero missing values**.

---

### 4. Fixed 150-Stock Master Universe

A fixed random master universe is selected with a constant seed to make the experiment reproducible.

**Execution**

```python
returns = load_data("data/returns.csv")

master_returns = select_master_universe(
    returns,
    n_assets=150,
    random_seed=RANDOM_SEED
)

save_data(master_returns, "data/master_returns_150.csv")
```

**Result**

```text
Master universe shape: (2714, 150)
```

Nested subsets are then used:

```text
N = 25  -> first 25 assets
N = 50  -> first 50 assets
N = 100 -> first 100 assets
N = 150 -> all 150 assets
```

This controls the asset universe while varying portfolio dimensionality.

---

### 5. Initial Controlled Example: N = 25, T = 126

The first test uses:

$$N = 25, \qquad T = 126$$

Therefore:

$$N/T = 25/126 \approx 0.198$$

---

### 6. Sample Covariance vs Ledoit-Wolf

The sample covariance matrix is:

$$\hat{\Sigma}_{sample} = \frac{1}{T-1}(R-\bar R)^T(R-\bar R)$$

Ledoit-Wolf shrinks the sample estimate toward a structured target:

$$\Sigma_{LW} = (1-\delta)\Sigma_{sample} + \delta F$$

**Execution**

```python
asset_returns = returns.iloc[:, :N]
window = asset_returns.iloc[:T]

cov_sample = sample_covariance(window)
cov_lw, shrinkage = ledoit_wolf_covariance(window)

sample_kappa, sample_min, sample_max = condition_number(cov_sample)
lw_kappa, lw_min, lw_max = condition_number(cov_lw)
```

**Result**

| Metric | Sample Covariance | Ledoit-Wolf Covariance |
|---|---:|---:|
| Condition number | 35.832950808515314 | 18.958990411343287 |
| Min eigenvalue | 0.00011430884297188795 | 0.00018644380209135653 |
| Max eigenvalue | 0.004096023147189962 | 0.0035347862561044135 |

```text
LW shrinkage intensity: 0.15234270071318765
```

**Interpretation**

Ledoit-Wolf applies approximately **15.2% shrinkage** and reduces the condition number from:

$$35.83 \rightarrow 18.96$$

an improvement of roughly **47%**.

The smallest eigenvalue increases while the largest decreases, compressing the eigenvalue spectrum and improving numerical stability.

---

### 7. GMV Portfolio Construction

The unconstrained Global Minimum Variance portfolio is:

$$w_{GMV} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}}$$

The implementation later uses the Moore-Penrose pseudoinverse to safely handle singular sample covariance matrices.

**Result for N = 25, T = 126**

<details>
<summary>Sample GMV weight statistics</summary>

```text
count    25.000000
mean      0.040000
std       0.082116
min      -0.044712
25%      -0.024384
50%       0.020634
75%       0.078893
max       0.290927
Sum: 1.0
Sample HHI: 0.2018310086878118
```
</details>

<details>
<summary>Ledoit-Wolf GMV weight statistics</summary>

```text
count    25.000000
mean      0.040000
std       0.063793
min      -0.037463
25%      -0.006688
50%       0.027806
75%       0.067051
max       0.223690
Sum: 1.0
LW HHI: 0.13766950321258073
```
</details>

```text
Equal-weight HHI: 0.04
```

The Ledoit-Wolf portfolio is substantially less concentrated.

Weight concentration is measured using:

$$HHI = \sum_i w_i^2$$

---

### 8. Out-of-Sample Realized Volatility

The GMV portfolios are held over the next 21 trading days, unseen during covariance estimation.

**Result**

| Portfolio | Realized Volatility |
|---|---:|
| Sample | 0.12142140435491272 |
| Ledoit-Wolf | 0.11809967346720758 |

In the first holding window:

$$12.14\% \rightarrow 11.81\%$$

in favor of Ledoit-Wolf.

---

### 9. Risk Forecast Error

Predicted annualized portfolio volatility is:

$$\hat\sigma_p = \sqrt{w^T\hat\Sigma w}\sqrt{252}$$

Forecast error is:

$$|\hat\sigma_p - \sigma_{realized}|$$

**Result**

| | Predicted Vol | Realized Vol | Forecast Error |
|---|---:|---:|---:|
| Sample | 0.11514691156929754 | 0.12142140435491272 | 0.006274492785615177 |
| Ledoit-Wolf | 0.12199865781287852 | 0.11809967346720758 | 0.0038989843456709444 |

Ledoit-Wolf produces the smaller risk-forecast error in this example.

---

### 10. Single-Window Sharpe

Using the next 21-day holding period:

```text
Sample Sharpe: 3.570187800651217
LW Sharpe: 3.7998531874691492
```

These values are not treated as final Sharpe estimates because a 21-day annualized Sharpe is extremely noisy. The final project uses a stitched full out-of-sample return series instead (see [Section 19](#19-full-out-of-sample-sharpe)).

---

### 11. Turnover

Turnover between consecutive rebalances is measured as:

$$Turnover_t = \frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|$$

**Result**

```text
Sample turnover: 0.2665137494791514
LW turnover: 0.1755618539998135
```

Ledoit-Wolf therefore produces more stable portfolio weights in this example.

---

### 12. Walk-Forward Experiment for N = 25, T = 126

The experiment is then repeated through the full historical sample with a 21-day rebalance frequency.

**Execution**

```python
results = run_walk_forward(returns, N=25, T=126, rebalance_frequency=21)
results.to_csv("Results/results_N25_T126.csv", index=False)
```

**Result** — shape `(123, 14)`

| Metric | Sample | Ledoit-Wolf |
|---|---:|---:|
| Condition number | 65.893392 | 22.319895 |
| HHI | 0.214795 | 0.120831 |
| Realized vol | 0.154275 | 0.146288 |
| Forecast error | 0.046116 | 0.035627 |
| Sharpe | 1.465808 | 1.654254 |
| Turnover | 0.316599 | 0.197058 |

```text
lw_shrinkage: 0.192224
```

The mechanism is coherent:

$$\text{Shrinkage} \rightarrow \text{better conditioning} \rightarrow \text{less concentrated weights} \rightarrow \text{lower turnover} \rightarrow \text{better out-of-sample risk behavior}$$

---

### 13. Full 16-Setting Experiment

The full grid is:

```text
N = 25, 50, 100, 150
T = 126, 252, 504, 756
```

**Execution**

```python
results = run_experiment_grid(
    returns=returns,
    asset_counts=ASSET_COUNTS,
    lookback_windows=LOOKBACK_WINDOWS,
    rebalance_frequency=REBALANCE_FREQUENCY,
)

results.to_csv("Results/all_experiment_results.csv", index=False)
```

**Result**

```text
Final shape: (1752, 17)
```

The experiment contains **1,752 walk-forward rebalance observations** across the 16 configurations.

---

### 14. Grid Summary

The raw rebalance results are averaged within each $(N, T)$ configuration. The full summary is saved as `Results/grid_summary.csv`.

**Selected examples**

| N | T | N/T | Sample Condition | LW Condition | Sample HHI | LW HHI | Sample Realized Vol | LW Realized Vol | Sample Turnover | LW Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 756 | 0.033 | 28.91 | 24.04 | 0.135 | 0.124 | 0.1503 | 0.1495 | 0.0533 | 0.0489 |
| 50 | 252 | 0.198 | 110.35 | 58.66 | 0.142 | 0.105 | 0.1258 | 0.1196 | 0.2591 | 0.1905 |
| 100 | 126 | 0.794 | 5028.45 | 137.62 | 0.505 | 0.079 | 0.2047 | 0.1134 | 2.4495 | 0.5035 |
| 150 | 126 | 1.190 | inf | 201.85 | 1184.63 | 0.067 | 4.1684 | 0.1087 | 104.92 | 0.6222 |

---

### 15. Singular Sample Covariance at N > T

For $N = 150,\ T = 126$, we have $N > T$.

For a centered return matrix:

$$rank(\hat\Sigma) \le T - 1 \implies rank(\hat\Sigma) \le 125 < 150$$

so the sample covariance must be rank-deficient.

**Corrected result**

| N | T | N/T | Sample Condition | LW Condition |
|---:|---:|---:|---:|---:|
| 150 | 126 | 1.190476 | inf | 201.845906 |

The GMV implementation therefore uses the **Moore-Penrose pseudoinverse** rather than the ordinary matrix inverse.

> Even with the pseudoinverse, the sample-based portfolio remains extremely unstable — this is itself an important result rather than a numerical artifact to hide.

---

### 16. Ledoit-Wolf Advantage Metrics

For metrics where lower is better:

$$\text{LW Advantage} = \frac{Sample - LW}{Sample}$$

This is calculated for:

- Weight concentration
- Realized volatility
- Risk forecast error
- Turnover

Corresponding plots:

```text
figures/hhi_advantage_vs_nt.png
figures/realized_vol_advantage_vs_nt.png
figures/forecast_error_advantage_vs_nt.png
figures/turnover_advantage_vs_nt.png
```

They visually show that the Ledoit-Wolf advantage rises strongly as $N/T$ increases.

---

### 17. Relationship Between N/T and Ledoit-Wolf Advantage

The relationship is quantified using:

$$\rho = Corr(N/T, \text{LW Advantage})$$

and:

$$\text{LW Advantage} = \alpha + \beta(N/T) + \epsilon$$

The hypothesis corresponds to $\beta > 0$.

**Results**

| Metric | Correlation | p-value | Slope | R² |
|---|---:|---:|---:|---:|
| HHI advantage | 0.958888 | 4.77e-09 | 0.8476 | 0.9195 |
| Realized-vol advantage | 0.951228 | 1.55e-08 | 0.7583 | 0.9048 |
| Forecast-error advantage | 0.982896 | 1.10e-11 | 0.9084 | 0.9661 |
| Turnover advantage | 0.968217 | 8.08e-10 | 0.8196 | 0.9374 |

**Interpretation**

All four relationships are strongly positive:

- HHI advantage: $r = 0.959,\ R^2 = 0.919$
- Realized-volatility advantage: $r = 0.951,\ R^2 = 0.905$
- Forecast-error advantage: $r = 0.983,\ R^2 = 0.966$
- Turnover advantage: $r = 0.968,\ R^2 = 0.937$

The strongest relationship is for risk forecast error ($R^2 = 0.966$), meaning $N/T$ explains approximately **96.6% of the variation** in relative forecast-error improvement across the 16 experimental configurations.

> Because all configurations share the same underlying market history and nested asset universe, the very small p-values should be interpreted as evidence *within this experimental grid*, not as independent population-level causal evidence.

---

### 18. Condition Number Analysis

Condition number is used as the mechanism metric explaining downstream instability. A separate analysis fits:

$$\log(\kappa) = \alpha + \beta(N/T) + \epsilon$$

for finite covariance cases. Corresponding plot: `figures/condition_number_vs_nt.png`

**Results**

| | Slope | R² | p-value |
|---|---:|---:|---:|
| Sample covariance | 6.3391851764737694 | 0.8356075941588631 | 1.8763412401340119e-06 |
| Ledoit-Wolf | 2.3942663818903616 | 0.282215009976189 | 0.04156876339419387 |

```text
LW CONDITIONING ADVANTAGE
Correlation: 0.9210306729917377
Slope: 1.0846034524311488
R-squared: 0.8482975005916128
p-value: 1.105993597897243e-06
```

**Interpretation**

Sample-covariance conditioning deteriorates much faster ($\beta_{sample} = 6.34$) than Ledoit-Wolf ($\beta_{LW} = 2.39$).

The Ledoit-Wolf conditioning advantage itself has $r = 0.921,\ R^2 = 0.848$.

At the extreme $N/T = 1.19$ case:

$$\kappa_{sample} = \infty \qquad \text{while} \qquad \kappa_{LW} \approx 201.85$$

This provides the mechanism behind the portfolio results.

---

### 19. Full Out-of-Sample Sharpe

Rather than averaging noisy 21-day Sharpe ratios, each configuration's daily out-of-sample portfolio returns are stitched together and a single annualized Sharpe is calculated:

$$Sharpe = \frac{\bar r}{\sigma_r}\sqrt{252}$$

Daily return series: `Results/oos_returns/` · Final table: `Results/full_oos_sharpe.csv`

**Results**

| N | T | N/T | Sample Sharpe | LW Sharpe | Difference |
|---:|---:|---:|---:|---:|---:|
| 25 | 756 | 0.033 | 1.268 | 1.276 | 0.008 |
| 25 | 504 | 0.050 | 1.235 | 1.272 | 0.037 |
| 50 | 756 | 0.066 | 1.665 | 1.671 | 0.007 |
| 25 | 252 | 0.099 | 1.308 | 1.374 | 0.066 |
| 50 | 504 | 0.099 | 1.502 | 1.554 | 0.053 |
| 100 | 756 | 0.132 | 1.306 | 1.331 | 0.024 |
| 50 | 252 | 0.198 | 1.413 | 1.518 | 0.105 |
| 25 | 126 | 0.198 | 1.111 | 1.163 | 0.052 |
| 150 | 756 | 0.198 | 1.480 | 1.531 | 0.051 |
| 100 | 504 | 0.198 | 1.318 | 1.392 | 0.074 |
| 150 | 504 | 0.298 | 1.304 | 1.470 | 0.166 |
| 50 | 126 | 0.397 | 1.210 | 1.400 | 0.191 |
| 100 | 252 | 0.397 | 1.298 | 1.454 | 0.156 |
| 150 | 252 | 0.595 | 1.192 | 1.452 | 0.260 |
| 100 | 126 | 0.794 | 0.812 | 1.301 | 0.489 |
| 150 | 126 | 1.190 | -0.594 | 1.303 | **1.898** |

The Sharpe advantage is:

$$\Delta Sharpe = Sharpe_{LW} - Sharpe_{sample}$$

**Relationship with N/T**

```text
Sharpe advantage correlation: 0.8908669240268107
Correlation p-value: 3.6881840671942713e-06
Slope: 1.3103228744940305
R-squared: 0.7936438763249907
Regression p-value: 3.6881840671943234e-06
```

**Interpretation**

$$N/T \uparrow \Rightarrow \text{LW Sharpe advantage} \uparrow \qquad \text{with } r = 0.891,\ R^2 = 0.794$$

At $N/T > 1$, the sample-covariance GMV breaks down severely, producing a negative full out-of-sample Sharpe, while the Ledoit-Wolf portfolio remains stable.

---

## 20. Main Empirical Finding

Across the controlled $N/T$ grid, Ledoit-Wolf's advantage increased strongly as $N/T$ rose.

**Observed correlations with N/T:**

| Advantage Type | Correlation |
|---|---:|
| Weight concentration | 0.959 |
| Turnover | 0.968 |
| Realized-volatility | 0.951 |
| Risk-forecast-error | 0.983 |
| Conditioning | 0.921 |
| Full OOS Sharpe | 0.891 |

The results consistently support the original hypothesis.

---

## 21. Mechanism

The experiment supports the following mechanism:

$$N/T \uparrow \rightarrow \text{sample covariance estimation becomes noisier} \rightarrow \kappa(\Sigma_{sample}) \uparrow \rightarrow \Sigma^{-1}\text{ becomes unstable} \rightarrow \text{GMV weights become extreme} \rightarrow \text{concentration and turnover increase} \rightarrow \text{out-of-sample risk performance deteriorates}$$

Ledoit-Wolf shrinkage reduces the amplification of noisy covariance directions and therefore increasingly helps as the estimation problem becomes high-dimensional.

---

## 22. Final Conclusion

> **Across the controlled $N/T$ grid, Ledoit-Wolf's advantage increased strongly as $N/T$ rose. Correlations were approximately 0.95–0.98 for portfolio concentration, turnover, realized volatility, and risk-forecast error, 0.92 for covariance-conditioning advantage, and 0.89 for full out-of-sample Sharpe improvement.**

The results therefore support the hypothesis that shrinkage becomes more valuable as the number of assets grows relative to the amount of available historical data.

The strongest failure occurs when $N > T$, where the sample covariance becomes mathematically rank-deficient. In the $N = 150,\ T = 126$ configuration, the sample condition number becomes infinite and the sample-GMV portfolio becomes severely unstable, whereas Ledoit-Wolf remains numerically well-conditioned enough to construct a stable portfolio.

---

## 23. Important Limitations

1. The project starts from a current NIFTY 500 constituent universe, so historical tests may contain **survivorship bias**.
2. The 150-stock master universe is fixed using a reproducible random seed; results could be tested across multiple asset-universe draws as a robustness extension.
3. The 16 $(N, T)$ configurations share the same market history and nested assets, so regression p-values are descriptive evidence within the experimental design rather than independent causal evidence.
4. The GMV portfolio is unconstrained and can contain short positions.
5. Transaction costs are not deducted from Sharpe in the current core experiment, although turnover is explicitly measured.
6. The $N > T$ sample-covariance case uses the Moore-Penrose pseudoinverse so the portfolio calculation can continue; the singularity itself remains an important experimental result.

---

## 24. Main Execution Order

From a fresh clone with data access:

```bash
pip install -r requirements.txt
```

Then execute the workflow in this order:

| Step | Script | Purpose |
|:---:|---|---|
| 1 | `main.py` | Load NIFTY 500 symbols, download prices, filter complete assets, generate returns, create fixed 150-stock universe, run walk-forward experiments, save grid results |
| 2 | `plot_results.py` | Calculate relative LW advantages; save HHI, turnover, realized-volatility, and forecast-error plots |
| 3 | `analyze_relationship.py` | Pearson correlations; simple regressions of LW advantage on N/T; save `nt_relationship_results.csv` |
| 4 | `analyze_conditioning.py` | Condition-number analysis; identify singular sample covariance cases; save condition-number plot and CSV |
| 5 | `analyze_sharpe.py` | Stitch daily out-of-sample returns; compute full-period Sharpe for all 16 settings; save `full_oos_sharpe.csv` and daily OOS return files |

---

## 25. Key Output Files

| File | Description |
|---|---|
| `Results/all_experiment_results.csv` | Raw walk-forward observations across all configurations |
| `Results/grid_summary.csv` | One aggregated row for each $(N, T)$ configuration |
| `Results/nt_relationship_results.csv` | Correlation and regression results for the main hypothesis |
| `Results/condition_number_analysis.csv` | Condition-number analysis and finite conditioning advantages |
| `Results/full_oos_sharpe.csv` | Full stitched out-of-sample Sharpe comparison |
| `Results/oos_returns/` | Daily out-of-sample Sample and Ledoit-Wolf GMV return series |
| `figures/` | Final visual evidence for the $N/T$ hypothesis |
