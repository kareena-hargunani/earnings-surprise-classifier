# Earnings Surprise Classifier

A logistic regression model that predicts whether a stock will experience a significant abnormal return following an earnings announcement, using only information available before the announcement date.

## Motivation

Standard event studies (Fama-French methodology) take semi-strong-form market efficiency as a *maintained assumption*. Then, it follows that the price reaction around a public disclosure is the market's response to the new information that the disclosure makes available to the public. This project does not take semi-strong EMH as a given assumption. This project asks the core research question: Does semi-strong EMH hold with respect to pre-announcement public information for earnings events, and can that information predict, out-of-sample, which future announcements will produce large abnormal reactions?

The literature documents cases where public information is only partially or gradually incorporated into price: post-earnings-announcement drift, driven in part by investors underestimating the persistence of earnings shocks (Bernard & Thomas, 1989, 1990). Motivated by this, using only information that was public *before* an earnings announcement (prior surprise history, momentum, volatility, macro regime), can we predict which announcements will produce a large positive abnormal return (CAR > 3%) *on the announcement itself*?

Under strict semi-strong EMH, the answer should be no because all information in these features should already be reflected in the pre-announcement price, leaving the announcement-day reaction driven only by new information in the release. Finding statistically significant, out-of-sample predictive power is evidence that some pre-announcement public information is not being fully priced in before the event rather than a claim that markets are broadly inefficient.

**Possible applications.** If pre-announcement features have predictive power over announcement-day reaction, it implies that part of a stock's reaction on an earnings date may be attributable to the market catching up on already-public information rather than to the new content of the announcement itself. In a standard event study model, the core research question is whether abnormal returns can be attributable to new information revealed in the disclosure. This project doesn't attempt that decomposition, which would require modeling the continuous, signed CAR rather than a binary threshold, and validating by ticker rather than in aggregate, but the underlying question it tests regarding announcement-date reaction is the same.

For example, if the EPS-surprise-history features (`eps_surprise_mean_4q`, `eps_beat_streak`, etc.) showed statistically significant, out-of-sample predictive power, that would be consistent with the market underestimating the persistence of earnings shocks (Bernard & Thomas, 1990) — not post-earnings-announcement drift itself (Bernard & Thomas, 1989), which is about predictability *after* a surprise is already revealed, rather than predicting the *next* announcement's reaction from *prior* surprise history. Such a finding would suggest a standard event study attributing the full announcement-date CAR to the disclosure itself is overstating that disclosure's specific contribution, since part of the reaction would reflect the market catching up on an already-public, predictable pattern rather than anything new in this particular announcement.

The direct way to actually correct an event study's benchmark for this uses what this project already has: characteristic-based residualization. Instead of building a new factor, model the continuous, signed CAR directly as a function of pre-announcement features (a regression instead of the binary classifier used here), producing a predicted `car_hat` for each event, and treat `actual_car - car_hat` as the disclosure-specific abnormal return. This doesn't require touching the label-construction pipeline or building new factor portfolios — it's a second-stage adjustment on top of the existing FF3-based CAR, reusing the same feature set and modeling approach already built here.

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
| Fundamental | P/E ratio |
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

**Why Fama-French 3-factor abnormal returns, not size-decile matching (Bernard & Thomas)?**
Bernard & Thomas (1989, 1990) benchmark against the equally-weighted return of the firm's NYSE/AMEX size-decile portfolio — a nonparametric approach standard in the earnings literature of the late 1980s, before the Fama-French factors existed. This project instead follows the Brown & Warner (1985) estimation-window/market-model tradition, extended to three factors: fit firm-specific factor loadings (α, β_mkt, β_smb, β_hml) over the pre-event estimation window, then use them to compute expected returns. This is now the standard approach in event-study work, and it additionally adjusts for value (HML) exposure, which size-decile matching ignores. It's also better suited to this project's S&P 500 universe — large, liquid, stable-beta names, where firm-specific beta estimation is reasonably precise — than it would be for the smaller, thinner NYSE/AMEX names in B&T's original sample. The trade-off: fitting four coefficients per event introduces estimation-error risk that a static size-decile assignment avoids by construction.

**Why L1 (Lasso), not Elastic Net, given known multicollinearity?**
Several features are deliberately correlated by construction — a raw feature and its own polynomial or interaction transform (e.g., `ret_5d` and `ret_5d²`). Lasso is known to make an arbitrary choice among correlated predictors (see the caveats in `plot_feature_importance` and `plot_correlation_heatmap`), and the standard fix is Elastic Net, which adds an L2 term that shrinks correlated coefficients toward each other instead of zeroing one out. This project keeps pure L1 for three reasons. First, the goal here is identifying which specific, named mechanisms (EPS-surprise persistence, momentum-regime dependence, residual momentum) survive out-of-sample scrutiny — a sparse result is a cleaner, more citable answer than Elastic Net's grouped output of many small, nonzero coefficients. Second, most of the correlation is between a variable and its own transform, not between independently-motivated, redundant proxies — the sharper version of the arbitrary-selection problem is reserved for near-duplicate features, which is why the genuinely redundant macro pair (`cfnai` vs. `yield_spread`) was resolved by removing one outright rather than by changing the penalty. Third, walk-forward folds have limited training data, especially early folds (as few as 8 quarters, ~25-35% positive rate) — Elastic Net tends to keep more coefficients nonzero, adding effective parameters the data may not support, and its extra hyperparameter (`l1_ratio`) would need to be tuned jointly with `C`, doubling the CV search space. The trade-off: feature-importance results still need the caveat already documented in the code — a surviving coefficient means "this feature or its correlated counterparts matter," not "this feature specifically and only this feature."

## Known Limitations

1. **Survivorship bias**: Uses current S&P 500 constituents. Companies that were delisted or removed are excluded, biasing the sample toward survivors.
2. **Concept drift**: Return anomalies tend to lose predictive power once documented. McLean & Pontiff (2016) show that across 97 documented cross-sectional return predictors, returns are 26% lower out-of-sample and 58% lower post-publication. Thus, `eps_surprise_mean_4q`, `eps_beat_streak`, and `eps_surprise_trend` may have measurably less forecasting power in more recent quarters than in earlier ones. I test this with `subperiod_analysis` and `plot_performance_over_time`. A downward trend in PR-AUC across regimes would be evidence of the decay in return anomalies' predictive power.
3. **Unknown future shocks**: No model can account for unscheduled macro events in the prediction window (COVID, SVB, etc.).
4. **Data quality**: yfinance EPS estimates are imperfect. WRDS/IBES would provide cleaner labels.
5. **Zero-positive test quarters**: PR-AUC and ROC-AUC are undefined when a test quarter has no CAR > 3% events, so `run_walk_forward` skips those folds entirely rather than fitting a model that can't be scored on those metrics. This is more likely in a small-universe run (e.g., the `N_TICKERS=30` development configuration) than in the full S&P 500 run. It means the reported aggregate PR-AUC is implicitly conditioned on quarters with at least one large positive surprise — the function discloses at runtime whether any quarters were actually skipped (see `results["skipped_quarters"]`), so this caveat should be read as "the mechanism exists," not as evidence it affected any specific run's results.

## Setup

```bash
pip install yfinance pandas-datareader pandas numpy scikit-learn statsmodels lxml matplotlib seaborn scipy
```

## Usage

```bash
# Run full analysis (set N_TICKERS to a small number in script for a development run)
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
