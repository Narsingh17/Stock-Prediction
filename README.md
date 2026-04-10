# 📈 Apple (AAPL) Stock Analysis
### Can We Predict the Next-Day Direction Better Than Guessing the Majority Class?

**University-Level Data Science Course Project**  
**Notebook:** `AAPL_Stock_Analysis.ipynb`  
**Data Period:** January 2014 – January 2024  
**Language:** Python 3 | Jupyter Notebook

---

## Research Question

> **Can we predict whether Apple stock will go UP or DOWN the next trading day better than always guessing the majority class?**

- **H₀ (Null):** Lagged return features carry no predictive power — model accuracy ≈ majority class baseline.
- **H₁ (Alternative):** Lagged return features provide signal — model accuracy > majority class baseline.
- **Success benchmark:** Beat the majority class accuracy (% of days AAPL went UP) on the held-out test set.

---

## Project Structure

```
AAPL_Stock_Analysis.ipynb   ← Main notebook (run top-to-bottom)
README.md                   ← This file
```

**Notebook sections:**

| Section | Title |
|---|---|
| 1 | Analytical Question & Hypothesis |
| 2 | Data Acquisition |
| 3 | Data Understanding |
| 4 | Data Cleaning & Preprocessing |
| 5 | Exploratory Data Analysis (EDA) |
| 6 | Dimension Reduction (6-Month Windows + PCA) |
| 7 | Statistical Modelling (Two Models) |
| 8 | Statistical Reasoning & Interpretation |
| 9 | Reproducibility & Professional Practice |
| 10 | Communication & Storytelling |

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

### 2. Launch Jupyter

```bash
jupyter notebook AAPL_Stock_Analysis.ipynb
# or in VS Code: open the .ipynb file directly
```

### 3. Run all cells top-to-bottom

- **Do not skip cells** — each section builds on the previous one.
- Data is fetched live from Yahoo Finance via `yfinance` for the 2014–2024 window.
- No external files or downloads are required — the notebook is fully self-contained.

### 4. Reproducibility guarantee

- `RANDOM_SEED = 42` is set at the top of Section 2 and passed to all stochastic functions (`np.random.seed`, all `random_state=` arguments, PCA, t-SNE).
- Train/test split is **chronological** — the first 80% of dates are training, the last 20% are test. No shuffling.
- Results are stable as long as Yahoo Finance does not revise the historical AAPL and `^GSPC` data for this period.

---

## Data Sources

| Dataset | Ticker | Source | Purpose |
|---|---|---|---|
| Apple Inc. daily OHLCV | `AAPL` | Yahoo Finance via `yfinance` | Primary analysis |
| S&P 500 Index daily OHLCV | `^GSPC` | Yahoo Finance via `yfinance` | Market benchmark (EDA) |

- **Period:** 2014-01-01 to 2024-01-01
- **Frequency:** Daily trading days (~252 per year, ~2,500 total)
- **Adjustment:** Auto-adjusted for stock splits and dividends

---

## Methodology Overview

### Feature Engineering (Modelling)
All features use **past data only** — no future information is used in any feature.

| Feature | Formula | Why |
|---|---|---|
| `r_lag1`, `r_lag2`, `r_lag3` | Previous 1–3 day returns | Recent price direction |
| `momentum` | Mean of last 5 daily returns | Short-term trend |
| `volatility` | Std of last 5 daily returns | Recent risk level |

### Target Variable
$$Y_t = \begin{cases} 1 & \text{if } r_{t+1} > 0 \quad (\text{UP tomorrow}) \\ 0 & \text{if } r_{t+1} \leq 0 \quad (\text{DOWN or FLAT}) \end{cases}$$

### High-Dimensional Dataset (Section 6)
The raw dataset is low-dimensional (5 columns). A high-dimensional dataset is constructed by:

1. Taking 10 years of daily Close prices
2. Splitting into **non-overlapping 6-month windows** (~126 trading days each) → ~20 windows
3. Each window → **1 row**; each daily price within the window → **1 feature**

| | Before Reshape | After Reshape |
|---|---|---|
| Rows | ~2,500 daily rows | **~20 six-month windows** |
| Columns | 1 (Close price) | **~120 daily prices per window** |

PCA reduces the ~120 features to 3 interpretable components.

### Two Prediction Models

| | Model 1 | Model 2 |
|---|---|---|
| Name | Low-Dimensional LR | High-Dimensional + PCA LR |
| Features | 5 (lags, momentum, volatility) | 20 lagged returns → 3 PCA components |
| Algorithm | Logistic Regression | Logistic Regression |
| Interpretability | High | Medium |

### Data Leakage Prevention
| Step | How Leakage Is Prevented |
|---|---|
| Lag features | Only past returns used — no future data |
| Train/test split | Chronological — future never in training |
| StandardScaler | Fitted on training data only |
| PCA | Fitted on training data only |

---

## Key Parameters

| Parameter | Value |
|---|---|
| Random seed | `42` |
| Train / test split | 80% / 20% (chronological) |
| Data period | 2014-01-01 to 2024-01-01 |
| PCA components | 3 |
| HD lags (Model 2) | 20 lagged daily returns |
| 6-month windows | ~20 rows × ~120 features |

---

## Outputs Generated

The notebook saves the following plots to the working directory:

| File | Description |
|---|---|
| `plot_price_trend.png` | Normalized AAPL vs S&P 500 price (2014–2024) |
| `plot_returns_dist.png` | AAPL daily return distribution |
| `plot_volatility.png` | 30-day rolling annualized volatility |
| `plot_corr_heatmap.png` | Pearson correlation heatmap of lagged returns |
| `plot_hd_windows.png` | All 6-month price window trajectories (high-dim dataset) |
| `plot_pca_scree.png` | PCA scree plot and cumulative variance |
| `plot_pca_loadings.png` | PC loadings — economic meaning of each component |
| `plot_pca_vs_tsne.png` | Side-by-side: PCA 2D vs t-SNE 2D on window data |
| `plot_coef_ld.png` | Logistic regression coefficients (Model 1) |
| `plot_confusion.png` | Confusion matrices for both models |
| `plot_comparison.png` | Accuracy bar chart: both models vs baseline |

---

## Limitations

- **No external factors** — earnings announcements, Federal Reserve decisions, news sentiment, and geopolitical events are not included.
- **Linear decision boundary** — logistic regression cannot capture non-linear patterns (momentum thresholds, regime switching).
- **Near-efficient markets** — by design, stock returns are hard to predict from price history alone.
- **6-month window PCA is illustrative** — ~20 rows is too few for reliable predictive modelling; it demonstrates the dimension reduction concept only.
- **Single train/test split** — a walk-forward (rolling) cross-validation would give more robust performance estimates.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Download historical stock data from Yahoo Finance |
| `pandas` | Data manipulation and time-series handling |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Statistical visualizations (heatmap, styling) |
| `scikit-learn` | Logistic Regression, PCA, t-SNE, StandardScaler, metrics |

Install all at once:
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

---

## Analytical Flow

```
QUESTION     Can we predict AAPL next-day direction > majority class baseline?
     |
DATA         10 years AAPL + S&P 500 from yfinance (2014-2024)
     |
CLEANING     No issues found — defensive pipeline applied to both datasets
     |
EDA          ~53% UP days (baseline); near-zero lag correlations; COVID/2022 vol spikes
     |
DIM REDUC    6-month windows → ~20 rows × ~120 features → PCA → 3 components
             PC1 = price level | PC2 = direction | PC3 = oscillation
             PCA vs t-SNE compared
     |
MODELS       Model 1: 5 features → Logistic Regression
             Model 2: 20 lags → PCA (3 PCs) → Logistic Regression
     |
RESULT       Both models evaluated vs majority class baseline (~53%)
     |
CONCLUSION   Price history alone provides limited predictive edge (near-efficient market)
```

---

## What We'd Explore With More Time

- Add macro features: VIX, Fed funds rate, earnings calendar
- Walk-forward (rolling) cross-validation instead of a single split
- Non-linear models: Random Forest, XGBoost — with the same leakage controls
- Test whether 6-month window PCA features add signal when combined with daily models

---

*Project completed as part of a university-level Data Science course.*
