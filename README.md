# Earnings Surprise Classifier

A logistic regression model that predicts whether a stock will experience a significant abnormal return following an earnings announcement, using only information available before the announcement date.

## Motivation

Standard event studies (Fama-French methodology) measure whether post-announcement returns are statistically abnormal *after the fact*. This project flips the question: can we predict *in advance* which earnings announcements will produce large abnormal returns, using publicly available pre-announcement features?

If markets are fully efficient (strong-form EMH), pre-announcement public information should have no predictive power. Finding non-trivial predictability — or understanding precisely *why* the model fails — is the core research question.

## Methodology

### Label Construction
Labels are built using the Fama-French 3-factor event study methodology:
1. Estimate the FF3 model on a pre-event estimation window (days −252 to −10)
2. Compute expected returns on the event window (days 0 to +1)
3. Label = 1 if cumulative abnormal return (CAR) > 3%, else 0

### Feature Groups
All features are computed strictly before the announcement date:

| Group | Features |
|---|---|
| EPS history | Prior quarter surprise, 4Q average, beat streak, surprise trend |
| Momentum | 5-day, 60-day, and 252-day pre-announcement returns |
| Volatility | Realized vol (30d, 60d), volatility ratio |
| Fundamental | Log market cap, P/E ratio |
| Macro regime | VIX (absolute + relative), yield spread, 10Y Treasury, recession indicator, market return |

Interaction terms and polynomial features are added for economically motivated nonlinear relationships before standardization.

### Model
- L1 (Lasso) logistic regression via `LogisticRegressionCV`
- Regularization parameter C tuned by inner cross-validation within each training fold
- StandardScaler fit on training fold only — never on test data

### Validation
- Walk-forward cross-validation with quarterly fold boundaries
- Expanding window (primary) and rolling window (comparison)
- Primary metric: Precision-Recall AUC (appropriate for imbalanced classes)
- Subperiod analysis across market regimes (Pre-COVID, COVID, Post-COVID, Rate Hike Cycle)

## Key Design Decisions

**Why Precision-Recall AUC, not accuracy?**
The positive class rate is ~25-35%. A model that always predicts 0 achieves 65-75% accuracy trivially. PR-AUC evaluates performance where it matters — on the positive class.

**Why walk-forward CV, not random split?**
Random splits leak future information into the training set. Walk-forward CV enforces temporal ordering, simulating realistic deployment conditions.

**Why standardize inside folds, not before?**
Fitting StandardScaler on the full dataset before CV leaks future distributional information (e.g., future mean and variance) into the training process. Scaling inside each fold uses only information available at the time of each training step.

**Why add interactions and polynomials before scaling?**
Scaling first and then squaring changes the meaning of polynomial terms. The correct order is: raw features → nonlinear transforms → scale → fit Lasso.

## Known Limitations

1. **Survivorship bias**: Uses current S&P 500 constituents. Companies that were delisted or removed are excluded, biasing the sample toward survivors.
2. **Cross-sectional dependence**: Same-date earnings events share common macro shocks. Standard errors are understated; effective sample size < row count.
3. **Concept drift**: Earnings momentum anomalies weaken as they are exploited by quant funds.
4. **Unknown future shocks**: No model can account for unscheduled macro events in the prediction window (COVID, SVB, etc.).
5. **Data quality**: yfinance EPS estimates are imperfect. WRDS/IBES would provide significantly cleaner labels.

## Setup

```bash
pip install yfinance pandas-datareader pandas numpy scikit-learn matplotlib seaborn scipy
```

## Usage

```bash
# Run full analysis (set N_TICKERS=30 in script for a development run)
python 01_run_analysis.py
```

## Project Structure

```
earnings-surprise-classifier/
├── src/
│   ├── data.py          # Data collection and caching
│   ├── features.py      # Label construction + feature engineering
│   └── model.py         # Walk-forward CV, evaluation, visualization
├── data/
│   ├── raw/             # Cached API pulls (CSV/parquet)
│   └── processed/       # Labeled events + feature matrix
├── report/              # Output charts and tables
├── 01_run_analysis.py   # Master script
└── README.md
```
