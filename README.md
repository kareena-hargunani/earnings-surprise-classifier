# Earnings Surprise Classifier

A logistic regression model that predicts whether a stock will experience a significant abnormal return following an earnings announcement, using only information available before the announcement date.

## Motivation

Standard event studies (Fama-French methodology) take semi-strong-form market efficiency as a *maintained assumption*. Then, it follows that the price reaction around a public disclosure is the market's response to the new information that the disclosure makes available to the public. This project does not take semi-strong EMH as a given assumption. This project asks the core research question: Does semi-strong EMH hold with respect to pre-announcement public information for earnings events, and can that information predict, out-of-sample, which future announcements will produce large abnormal reactions?

The literature documents cases where public information is only partially or gradually incorporated into price: post-earnings-announcement drift, driven in part by investors underestimating the persistence of earnings shocks (Bernard & Thomas, 1989, 1990). Motivated by this, using only information that was public *before* an earnings announcement (prior surprise history, momentum, volatility, macro regime), can we predict which announcements will produce a large positive abnormal return (CAR > 3%) *on the announcement itself*?

Under strict semi-strong EMH, the answer should be no because all information in these features should already be reflected in the pre-announcement price, leaving the announcement-day reaction driven only by new information in the release. Finding statistically significant, out-of-sample predictive power is evidence that some pre-announcement public information is not being fully priced in before the event rather than a claim that markets are broadly inefficient.

**Possible applications.** If pre-announcement features have predictive power over announcement-day reaction, it implies that part of a stock's reaction on an earnings date may be attributable to the market catching up on already-public information rather than to the new content of the announcement itself. In a standard event study model, the core research question is whether abnormal returns can be attributable to new information revealed in the disclosure. This project doesn't attempt that decomposition, which would require modeling the continuous, signed CAR rather than a binary threshold, and validating by ticker rather than in aggregate, but the underlying question it tests regarding announcement-date reaction is the same.

For example, if the EPS-surprise-history features (`eps_surprise_mean_4q`, `eps_beat_streak`, etc.) showed statistically significant, out-of-sample predictive power, that would be consistent with the market underestimating the persistence of earnings shocks (Bernard & Thomas, 1990) — not post-earnings-announcement drift itself (Bernard & Thomas, 1989), which is about predictability *after* a surprise is already revealed, rather than predicting the *next* announcement's reaction from *prior* surprise history. Such a finding would suggest a standard event study attributing the full announcement-date CAR to the disclosure itself is overstating that disclosure's specific contribution, since part of the reaction would reflect the market catching up on an already-public, predictable pattern rather than anything new in this particular announcement.

A direct way to actually correct an event study's benchmark is to model the continuous, signed CAR directly as a function of pre-announcement features (a regression instead of the binary classifier used here), producing a predicted `car_hat` for each event, and treat `actual_car - car_hat` as the disclosure-specific abnormal return. This doesn't require touching the label-construction pipeline or building new factor portfolios — it's a second-stage adjustment on top of the existing FF3-based CAR, reusing the same feature set and modeling approach already built here.

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

## Results

The model was evaluated out-of-sample across 33 quarterly folds on the full S&P 500 (503 tickers, 19,930 labeled events), spanning four distinct market regimes — Pre-COVID, COVID, Post-COVID, and the Rate Hike Cycle — from mid-2018 through mid-2026.

A naive classifier, which predicts a surprise at the ratio of true positive values to sample size, is **0.184**. This model's pooled PR-AUC, evaluated as one combined test over all 33 folds, is **0.226**, which performs better than the baseline. Averaging performance quarter by quarter instead of pooling everything together gives a mean PR-AUC of **0.273 ± 0.052**. Fold-by-fold, the model consistently beats its local baseline. Additionally, variance is small, indicating that the model's PR-AUC is consistent across folds. 

Breaking performance out by regime tells an important second part of the story. Raw PR-AUC isn't directly comparable across regimes, because each regime has its own different rate of large positive surprises to begin with. A period with more big surprises has a higher baseline. A "lift" ratio allows us to compare each regime's PR-AUC to *its own* baseline: Pre-COVID performs best at 1.65x its own baseline, followed by Post-COVID (1.48x), COVID (1.46x), and the Rate Hike Cycle (1.44x). Consistency across the lift ratio indicates that there is no sigificant regime-dependence.

By coefficient magnitude, the strongest predictors are volatility-based followed by macro-regime variables (interest rates, the VIX). The EPS-surprise-history and momentum interaction features also have a small, but non-zero effect on EPS surprise. 

This project aims to answer whether pre-announcement public information has any predictive power over earnings surprises. We find that the earning surprise classifier performs better than a naive one, indicating that some pre-announcement public information contains predictive power. 
However, the strongest driver, volatility, doesn't have a clean interpretation. During high volatility periods, there could be a rational risk premium that the FF3 benchmark doesn't fully capture. If the FF3 model understates the true expected return for volatile stocks, CAR would be inflated and therefore the incicdence of positive events would be inflated. Thus, high volatility periods could systematically bias surprise events upwards rather than containing true high predictive power. Macro regime factors and past quarter’s earnings surprise behavior also have non-zero predictive power, indicating that these factors are not fully priced into the market. 
Macro regime factors such as the 10Y treasury and VIX level could genuinely contain some additional predictive power, but may also capture the effect of structurally different β_mkt and β_hml than the fixed estimates produced during label construction. We tested this directly with a Chow test on each event's estimation window: 21.1% of events show a detected structural break, well above the ~5% expected by chance, and concentrated in COVID (35.4%) and Post-COVID (29.4%) versus Pre-COVID (15.9%). Comparing events with and without a detected break, `realized_vol_60d`, `realized_vol_30d`, `treasury_10y`, and `eps_beat_last_q` all retain — or show *stronger* — correlations with the label in the "clean" (no detected break) subsample. That's evidence against structural breaks being the primary driver of these coefficients, though it doesn't resolve whether the remaining relationship reflects a rational risk premium or genuine mispricing (see `README.md` Known Limitations #6 for the full breakdown).
Past quarter's earnings surprise is also significant, in line with Bernard & Thomas' finding that investors underestimate the persistence of EPS beat. Thus, investors do not raise their expecations of future earnings proportionally to the stock's earnings momentum. 
While I find positive, non-zero coefficients on volatility, macro regime, and EPS factors, the finding does not show that markets are inefficient. The Chow-test results argue against structural breaks being the primary explanation, but the coefficients could still reflect a real rational risk premium that FF3 doesn't measure. However, the results do raise caution in taking semi-strong EMH as a given assumption when predicting whether an earnings event alone causes an abnormal stock price reaction. 

**Robustness check: gradient boosted trees.** The Lasso model relies specified interaction and polynomial terms. As a robustness check, I ran the same walk-forward evaluation with an `XGBClassifier` which can discover interactions and nonlinearities automatically. Its hyperparameters (tree depth, learning rate) were tuned the same way as Lasso's `C`, via an inner `TimeSeriesSplit` on each training fold. This model yields a mean PR-AUC of 0.276 ± 0.049 while the Lasso model yields a mean PR-AUC of 0.273 ± 0.052. The performance of the models are well within fold-to-fold noise. The inner CV selected the shallowest tree depth in the search grid for almost every fold, meaning the model had the freedom to build deep, complex interaction structures and mostly chose not to. This suggests that the specified interaction terms capture close to what a fully flexible model finds on the same data. 

## Key Design Decisions

**Why Precision-Recall AUC, not accuracy?**
The positive class rate is ~25-35%. A model that always predicts 0 achieves 65-75% accuracy trivially. PR-AUC evaluates performance on the positive class.

**Why walk-forward CV, not random split?**
Random splits leak future information into the training set. Walk-forward CV enforces temporal ordering.

**Why standardize inside folds, not before?**
Fitting StandardScaler on the full dataset before CV leaks future distributional information (e.g., future mean and variance) into the training process. Scaling inside each fold uses only information available at the time of each training step.

**Why add interactions and polynomials before scaling?**
Scaling first and then squaring changes the meaning of polynomial terms because we are fitting Lasso. The correct order is: raw features → nonlinear transforms → scale → fit Lasso.

**Why Fama-French 3-factor abnormal returns, not CAPM (single-factor market model)?**
The standard event study model is the same form as CAPM. Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds" (*Journal of Financial Economics*, 33, 3-56), showed that size (SMB) and value (HML) carry additional, systematic explanatory power beyond the market factor alone, and FF3 is better at capturing the systematic variation in returns. This leaves a smaller, less noisy residual to interpret as "abnormal." In this project, the FF3 model is used to calculate the abnormal return that results in a binary classification of surprise(1) or not (0). 

**Why L1 (Lasso), not Elastic Net, given known multicollinearity?**
Several features are deliberately correlated by construction — a raw feature and its own polynomial or interaction transform (e.g., `ret_5d` and `ret_5d²`). Lasso is known to make an arbitrary choice among correlated predictors (see the caveats in `plot_feature_importance` and `plot_correlation_heatmap`), and the standard fix is Elastic Net, which adds an L2 term that shrinks correlated coefficients toward each other instead of zeroing one out. This project keeps pure L1 for three reasons. First, the goal here is identifying which specific, named mechanisms (EPS-surprise persistence, momentum-regime dependence, residual momentum) survive out-of-sample scrutiny, so a sparse result helps interpretability. Second, most of the correlation is between a variable and its own transformation rather than the inclusion of proxies for the same underlying market mechanism (i.e. macro regime). Third, walk-forward folds have limited training data (as few as 8 quarters), so fewer effective parameters reduces overfitting risk. We would also need to tune Elastic Net's extra hyperparameter (`l1_ratio`), which is the mixing ratio of the L1 and L2 terms. A caveat is that a non-zeroed out feature's correlated counterparts may matter rather than only the feature mattering.

## Known Limitations

1. **Survivorship bias**: Uses current S&P 500 constituents. Companies that were delisted or removed are excluded, biasing the sample toward survivors.
2. **Concept drift**: Return anomalies tend to lose predictive power once documented. McLean & Pontiff (2016) show that across 97 documented cross-sectional return predictors, returns are 26% lower out-of-sample and 58% lower post-publication. Thus, `eps_surprise_mean_4q`, `eps_beat_streak`, and `eps_surprise_trend` may have measurably less forecasting power in more recent quarters than in earlier ones. I test this with `subperiod_analysis` and `plot_performance_over_time`. A downward trend in PR-AUC across regimes would be evidence of the decay in return anomalies' predictive power.
3. **Unknown future shocks**: No model can account for unscheduled macro events in the prediction window (COVID, SVB, etc.).
4. **Data quality**: yfinance EPS estimates are imperfect. WRDS/IBES would provide cleaner labels.
5. **Zero-positive test quarters**: PR-AUC and ROC-AUC are undefined when a test quarter has no CAR > 3% events, so `run_walk_forward` skips those folds entirely rather than fitting a model that can't be scored on those metrics. This is more likely in a small-universe run (e.g., the `N_TICKERS=30` development configuration) than in the full S&P 500 run. It means the reported aggregate PR-AUC is implicitly conditioned on quarters with at least one large positive surprise — the function discloses at runtime whether any quarters were actually skipped (see `results["skipped_quarters"]`), so this caveat should be read as "the mechanism exists," not as evidence it affected any specific run's results.
6. **Structural breaks within the estimation window**: For each event, the FF3 model is estimated using a fixed estimation window. If there is a structural break during the estimation window, a firm's actual risk exposure would shift during the estimation window. The estimated beta blends the pre- and post-break exposure, misspecifying the "normal" return and adding noise to CAR.

  We tested this directly with a fixed-midpoint Chow test (`notebooks/_chow_test.py`): split each event's estimation window in half, re-estimate FF3 on each half, and test whether the factor loadings differ significantly across the two halves. Across all 19,930 events, **21.1%** show a detected break (p < 0.05) — well above the ~5% you'd expect by chance if beta were actually stable everywhere. Incidence of structural breaks are concentrated as follows: 35.4% in COVID, 29.4% in Post-COVID, versus 15.9% in Pre-COVID.

  We then used this to test whether volatility and macro-regime features' predictive power is driven by this instability: split events into "Chow-clean" (no detected break) and "Chow-flagged" (detected break) subsamples, and compared each feature's point-biserial correlation with the label in each. If beta instability were the main driver, the relationship should be *stronger* in the flagged subsample. Instead, `realized_vol_60d`, `realized_vol_30d`, `treasury_10y`, and `eps_beat_last_q` all show correlations that are as strong or stronger in the Chow-clean subsample, and some (`treasury_10y`, `eps_beat_last_q`) lose significance entirely in the flagged subsample. That's evidence against beta instability being the primary driver of these features' predictive power but does not address whether there is a real risk premium not captured by FF3.

  One discrepancy is that `eps_beat_last_q` shows a *negative* univariate correlation with the label here, versus a *positive* coefficient in the full multivariate Lasso fit in Results. This is not necessarily a contradiction since an univariate correlation and a coefficient conditional on 25 other features can differ in sign.

  Further refinement: this used a single fixed-midpoint Chow test, which only has power to detect a break near the middle of the window. Instead, a Quandt-Andrews (sup-Chow) test, which searches across a range of candidate break dates and takes the maximum test statistic, would give a more reliable incidence estimate.

## Setup

```bash
pip install yfinance pandas-datareader pandas numpy scikit-learn statsmodels lxml matplotlib seaborn scipy xgboost
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
