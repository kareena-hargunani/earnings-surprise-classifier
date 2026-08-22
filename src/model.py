"""
model.py
--------
Modeling and evaluation for the Earnings Surprise Classifier.

Implements walk-forward cross-validation with three model layers:

    Layer 1 — Baseline logistic regression (raw features only)
    Layer 2 — Logistic regression with monotonic transformations
    Layer 3 — Logistic regression with polynomial + interaction terms

Each layer uses L1 (Lasso) regularization with the regularization
parameter C tuned via inner cross-validation WITHIN each training fold.

CRITICAL IMPLEMENTATION NOTE:
    StandardScaler is fit on the TRAINING fold only and applied to the
    test fold. It is never fit on the test fold or the full dataset.
    This prevents future distributional information from leaking into
    the feature scaling — a subtle form of lookahead bias.

Walk-forward validation:
    - Expanding window (default): train on all quarters up to fold k,
      test on quarter k+1
    - Rolling window (optional): train on a fixed N most recent quarters,
      test on the next quarter. Use this to assess whether recency of
      training data matters — if rolling significantly outperforms expanding,
      it suggests concept drift in older data.

Evaluation metrics:
    - Precision-Recall AUC (primary) — appropriate for imbalanced classes
    - ROC-AUC (secondary)
    - Precision and Recall at fixed thresholds
    - Brier score — calibration quality

We deliberately do NOT report accuracy as a primary metric. With ~25-35%
positive class rate, a model that always predicts 0 achieves 65-75%
accuracy trivially. Precision-recall tells the real story.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
# Non-interactive backend: plt.show() becomes a no-op instead of requiring
# a display. This script always saves figures via save_path, so plt.show()
# calls only exist for interactive/notebook use — in a plain script or
# background process, the default GUI backend can crash rather than no-op.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Walk-Forward Cross-Validation ────────────────────────────────────────────

def make_quarterly_folds(
    df: pd.DataFrame,
    min_train_quarters: int = 8,
    rolling_window: int = None,
) -> list[tuple]:
    """
    Generate walk-forward train/test splits at the quarterly level.
    
    Each fold:
        - Training set: all events in quarters up to (and including) fold_q
        - Test set: all events in the NEXT quarter
    
    Parameters
    ----------
    df                  : feature DataFrame with 'earnings_date' column
    min_train_quarters  : minimum quarters in first training set
    rolling_window      : if set, use a rolling window of this many quarters
                          rather than an expanding window. Pass None for expanding.
    
    Returns a list of (train_idx, test_idx) tuples.
    
    WHY QUARTERLY FOLDS?
    Using calendar quarters as fold boundaries is more appropriate than
    arbitrary date splits because earnings seasons cluster by quarter.
    Using random date splits would create folds where adjacent earnings
    seasons straddle the train/test boundary, causing correlated observations
    to leak across folds.
    """
    df = df.copy()
    df["quarter"] = df["earnings_date"].dt.to_period("Q")
    quarters = sorted(df["quarter"].unique())
    
    if len(quarters) < min_train_quarters + 1:
        raise ValueError(
            f"Need at least {min_train_quarters + 1} quarters of data. "
            f"Got {len(quarters)}."
        )
    
    folds = []
    for i in range(min_train_quarters, len(quarters) - 1):
        test_q = quarters[i + 1]
        
        if rolling_window:
            train_start_q = quarters[max(0, i - rolling_window + 1)]
            train_mask = (df["quarter"] >= train_start_q) & (df["quarter"] <= quarters[i])
        else:
            train_mask = df["quarter"] <= quarters[i]
        
        test_mask = df["quarter"] == test_q
        
        train_idx = df[train_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx, str(test_q)))
    
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    rolling_window: int = None,
    C_values: list = None,
    random_state: int = 42,
) -> dict:
    """
    Run walk-forward cross-validation for all three model layers.
    
    For each fold:
        1. Split into train/test by quarter
        2. Impute missing values with training set median (fit on train only)
        3. Standardize features (fit on train only)
        4. Tune C via inner CV on training fold
        5. Fit final model and predict on test fold
        6. Record metrics
    
    Returns a dict with fold-level and aggregate metrics for each layer.
    """
    C_values = C_values or [0.001, 0.01, 0.1, 1.0, 10.0]
    
    df = df.copy().reset_index(drop=True)
    folds = make_quarterly_folds(df, rolling_window=rolling_window)
    
    print(f"Walk-forward CV: {len(folds)} test folds")
    print(f"Window type: {'Rolling (' + str(rolling_window) + ' quarters)' if rolling_window else 'Expanding'}")
    
    results = {
        "fold_metrics": [],
        "feature_importances": [],
        "predictions": [],
        "skipped_quarters": [],
    }

    for fold_num, (train_idx, test_idx, test_q) in enumerate(folds):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # Skip folds with no positive examples in test set — PR-AUC and
        # ROC-AUC are undefined with only one class present. Recorded in
        # results["skipped_quarters"] so the aggregate metrics below can
        # disclose whether any quarters were dropped, rather than silently
        # shrinking the reported fold count.
        if y_test.sum() == 0:
            results["skipped_quarters"].append(test_q)
            continue
        
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # Step 1: Impute missing values with TRAINING median
        # Fit imputer on training data only — never on test data
        train_medians = np.nanmedian(X_train, axis=0)
        X_train_imp = np.where(np.isnan(X_train), train_medians, X_train)
        X_test_imp = np.where(np.isnan(X_test), train_medians, X_test)
        
        # Step 2: Standardize using TRAINING statistics only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)  # transform, NOT fit_transform
        
        # Step 3: Fit L1 logistic regression with C tuned by inner CV
        # TimeSeriesSplit enforces that each inner validation block only
        # ever follows its training data in time — a plain integer `cv`
        # would default to StratifiedKFold, which (with shuffle=False)
        # still lets later chronological data train on earlier validation
        # blocks in some splits, leaking lookahead bias into C selection.
        model = LogisticRegressionCV(
            Cs=C_values,
            cv=TimeSeriesSplit(n_splits=5),
            penalty="l1",
            solver="saga",
            scoring="average_precision",
            max_iter=2000,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)
        
        # Step 4: Predict on test fold
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Step 5: Compute metrics
        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else np.nan
        brier = brier_score_loss(y_test, y_prob)
        
        fold_result = {
            "test_quarter": test_q,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_positive_test": int(y_test.sum()),
            "positive_rate_test": y_test.mean(),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "best_C": model.C_[0],
        }
        results["fold_metrics"].append(fold_result)
        
        # Store feature importances (non-zero coefficients after L1)
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": model.coef_[0],
            "test_quarter": test_q,
        })
        results["feature_importances"].append(coef_df)
        
        # Store predictions for later analysis
        pred_df = test_df[["ticker", "earnings_date", "label", "car"]].copy()
        pred_df["predicted_prob"] = y_prob
        pred_df["test_quarter"] = test_q
        results["predictions"].append(pred_df)
        
        if fold_num % 4 == 0:
            print(f"  Q{test_q}: PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, "
                  f"n_test={len(test_idx)}, best_C={model.C_[0]:.4f}")
    
    # Aggregate results
    fold_df = pd.DataFrame(results["fold_metrics"])
    results["fold_df"] = fold_df
    results["predictions_df"] = pd.concat(results["predictions"], ignore_index=True)
    results["importances_df"] = pd.concat(results["feature_importances"], ignore_index=True)
    
    print(f"\nAggregate results across {len(fold_df)} folds:")
    print(f"  Mean PR-AUC:    {fold_df['pr_auc'].mean():.3f} ± {fold_df['pr_auc'].std():.3f}")
    print(f"  Mean ROC-AUC:   {fold_df['roc_auc'].mean():.3f} ± {fold_df['roc_auc'].std():.3f}")
    print(f"  Mean Brier:     {fold_df['brier_score'].mean():.3f} ± {fold_df['brier_score'].std():.3f}")

    n_skipped = len(results["skipped_quarters"])
    if n_skipped == 0:
        print(f"\nNo quarters were skipped for having zero positive test examples — "
              f"the {len(fold_df)} folds above cover every quarter in the walk-forward "
              f"window, so this result is not a filtered subsample.")
    else:
        print(f"\n{n_skipped} quarter(s) skipped for having zero positive test examples "
              f"(PR-AUC/ROC-AUC are undefined with one class): {results['skipped_quarters']}. "
              f"Aggregate metrics above are computed only over the remaining {len(fold_df)} folds.")

    return results


# ── Subperiod Analysis ────────────────────────────────────────────────────────

def subperiod_analysis(results: dict) -> pd.DataFrame:
    """
    Break down walk-forward performance by market regime subperiod.
    
    Subperiods:
        Pre-COVID:   before 2020 Q1
        COVID:       2020 Q1 - 2020 Q4
        Post-COVID:  2021 Q1 - 2021 Q4
        Rate hikes:  2022 Q1 onwards
    
    This analysis directly tests whether cross-sectional dependence from
    macro shocks is clustering model errors in specific periods.
    If performance degrades sharply in any subperiod, that is evidence
    of regime-dependent relationships that the model fails to generalize across.
    """
    fold_df = results["fold_df"].copy()
    fold_df["test_quarter"] = pd.PeriodIndex(fold_df["test_quarter"], freq="Q")
    
    def assign_regime(q):
        if q < pd.Period("2020Q1", freq="Q"):
            return "Pre-COVID"
        elif q <= pd.Period("2020Q4", freq="Q"):
            return "COVID"
        elif q <= pd.Period("2021Q4", freq="Q"):
            return "Post-COVID"
        else:
            return "Rate Hike Cycle"
    
    fold_df["regime"] = fold_df["test_quarter"].apply(assign_regime)
    
    summary = fold_df.groupby("regime").agg(
        n_folds=("pr_auc", "count"),
        mean_pr_auc=("pr_auc", "mean"),
        std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_positive_rate=("positive_rate_test", "mean"),
    ).round(3)

    # Lift = PR-AUC relative to the regime's OWN no-skill baseline (its own
    # positive rate), not the raw PR-AUC. The no-skill baseline for PR-AUC
    # moves with the positive rate, and that rate varies by regime — so
    # comparing raw PR-AUC across regimes conflates "the model did better
    # here" with "this regime just had a higher base rate of large surprises
    # to begin with." Lift isolates the former.
    summary["lift"] = (summary["mean_pr_auc"] / summary["mean_positive_rate"]).round(2)

    print("\nSubperiod performance analysis:")
    print(summary.to_string())
    print("\nNOTE: Compare regimes by 'lift', not raw mean_pr_auc — the no-skill")
    print("baseline (positive rate) varies by regime, so raw PR-AUC conflates")
    print("model skill with how common large surprises were in that period.")
    print("Large differences in lift across regimes suggest regime-dependent")
    print("relationships; large differences in raw PR-AUC alone may not.")

    return summary


# ── Visualization ─────────────────────────────────────────────────────────────

NAVY = "#1B2A4A"
GOLD = "#C9A84C"
GRAY = "#888888"
LIGHT = "#F5F5F5"


def plot_pr_curve(results: dict, title: str = "Precision-Recall Curve", save_path: str = None):
    """
    Plot precision-recall curve using all out-of-sample predictions.
    
    We pool all walk-forward test fold predictions and compute a single
    PR curve. This gives an honest picture of out-of-sample performance
    across the full evaluation period.
    
    The no-skill baseline is a horizontal line at the positive class rate.
    A good model should substantially exceed this baseline.
    """
    preds = results["predictions_df"]
    y_true = preds["label"].values
    y_prob = preds["predicted_prob"].values
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor(LIGHT)
    
    ax.plot(recall, precision, color=NAVY, lw=2.5, label=f"Logistic Regression (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, color=GOLD, lw=1.5, linestyle="--", label=f"No-skill baseline ({baseline:.2f})")
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"PR-AUC: {pr_auc:.3f} (baseline: {baseline:.3f})")


def plot_feature_importance(results: dict, top_n: int = 15, save_path: str = None):
    """
    Plot mean absolute coefficient magnitude across walk-forward folds.
    
    L1 regularization zeros out features that don't contribute marginal
    predictive value. Features that survive regularization across multiple
    folds are the most consistent predictors.
    
    Important caveat in the write-up: due to multicollinearity, the model's
    choice of WHICH correlated feature to keep is somewhat arbitrary.
    Interpret importance as 'this feature or its correlated counterparts
    matter' rather than 'this specific feature and only this feature matters'.
    """
    imp_df = results["importances_df"]
    mean_imp = (
        imp_df.groupby("feature")["coefficient"]
        .apply(lambda x: np.abs(x).mean())
        .sort_values(ascending=True)
        .tail(top_n)
    )
    
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_facecolor(LIGHT)
    
    colors = [NAVY if v > 0 else GOLD for v in mean_imp.values]
    bars = ax.barh(mean_imp.index, mean_imp.values, color=NAVY, alpha=0.85)
    
    ax.set_xlabel("Mean |Coefficient| Across Walk-Forward Folds", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features by L1 Regularized Coefficient Magnitude\n"
        "(Features zeroed out by Lasso not shown)",
        fontsize=13, fontweight="bold", color=NAVY
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_performance_over_time(results: dict, save_path: str = None):
    """
    Plot walk-forward PR-AUC over time with regime shading.
    
    This is the key diagnostic chart for non-stationarity and
    cross-sectional dependence. Clusters of low-performance folds
    coinciding with macro stress periods (COVID, rate hike cycle)
    are evidence that the model struggles with regime shifts.
    """
    fold_df = results["fold_df"].copy()
    fold_df["test_quarter"] = pd.PeriodIndex(fold_df["test_quarter"], freq="Q")
    fold_df = fold_df.sort_values("test_quarter").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor(LIGHT)

    # Shade COVID period — only if a fold actually falls in that window.
    # Without this guard, when no fold overlaps COVID, both boundaries
    # fall back to 0 and axvspan(0, 0) draws a misleading zero-width
    # marker at the start of the chart, mislabeled "COVID period."
    covid_start = pd.Period("2020Q1", freq="Q")
    covid_end = pd.Period("2020Q4", freq="Q")
    covid_mask = (fold_df["test_quarter"] >= covid_start) & (fold_df["test_quarter"] <= covid_end)
    if covid_mask.any():
        ax.axvspan(
            fold_df[covid_mask].index.min(),
            fold_df[covid_mask].index.max(),
            alpha=0.15, color="red", label="COVID period"
        )

    x = range(len(fold_df))
    ax.plot(x, fold_df["pr_auc"], color=NAVY, lw=2, marker="o", ms=5, label="PR-AUC per fold")
    ax.axhline(fold_df["positive_rate_test"].mean(), color=GOLD, lw=1.5,
               linestyle="--", label="Avg no-skill baseline")
    ax.axhline(fold_df["pr_auc"].mean(), color=GRAY, lw=1.5,
               linestyle=":", label=f"Mean PR-AUC ({fold_df['pr_auc'].mean():.3f})")
    
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(q) for q in fold_df["test_quarter"].iloc[::2]], rotation=45, ha="right")
    ax.set_ylabel("PR-AUC", fontsize=11)
    ax.set_title("Walk-Forward PR-AUC Over Time\n(Clusters of low performance → regime-dependent relationships)",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str], save_path: str = None):
    """
    Plot a correlation heatmap of the feature matrix.
    
    This is a diagnostic step that should be run BEFORE modeling to:
        1. Identify highly correlated feature pairs
        2. Understand where multicollinearity will inflate standard errors
        3. Inform which features to expect Lasso to zero out
    
    Features with |correlation| > 0.7 are candidates for being
    partially redundant — Lasso will typically keep one and zero out
    the other, but the choice is somewhat arbitrary.
    """
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=False,
        fmt=".2f",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title(
        "Feature Correlation Matrix\n"
        "|corr| > 0.7 suggests multicollinearity — Lasso will handle redundancy",
        fontsize=13, fontweight="bold", color=NAVY, pad=15
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    # Print highly correlated pairs
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    
    if high_corr:
        print("\nHighly correlated feature pairs (|r| > 0.7):")
        for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {a} × {b}: r = {r:.3f}")
    else:
        print("No feature pairs with |r| > 0.7.")


if __name__ == "__main__":
    print("model.py loaded successfully.")
    print("Key functions:")
    print("  make_quarterly_folds() — generate walk-forward train/test splits")
    print("  run_walk_forward()     — full CV pipeline with scaling inside folds")
    print("  subperiod_analysis()   — performance by market regime")
    print("  plot_pr_curve()        — precision-recall curve")
    print("  plot_feature_importance() — L1 coefficient magnitudes")
    print("  plot_performance_over_time() — PR-AUC across folds")
    print("  plot_correlation_heatmap()   — feature multicollinearity diagnostic")
