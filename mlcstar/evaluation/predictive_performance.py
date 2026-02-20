# predictive_performance.py
"""
Predictive performance evaluation for mlcstar EBM models.

Contains:
- TimeMetricResult: dataclass for storing per-timepoint metrics
- generate_time_thresholds: generate evaluation time grid
- format_step_label: human-readable time labels
- plot_time_metrics: AUROC/AUPRC over time plots
- run_eval_ebm: evaluate a trained EBM at a single masking point
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass

from mlcstar.utils import cfg, logger, save_figure
from mlcstar.data.datasets import AggregatedDS
from mlcstar.evaluation.utils import (
    calculate_roc_auc_ci, calculate_average_precision_ci,
    time_to_step, step_to_time,
)
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class TimeMetricResult:
    """Container for time-dependent evaluation results."""
    time_min: float
    time_hours: float
    time_days: float
    censor_step: int
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    n_samples: int
    n_positive: int


# ============================================================================
# TIME THRESHOLD UTILITIES
# ============================================================================

def generate_time_thresholds(max_days=30, cut_hours=72, step_hours=1, step_days=1):
    """Generate list of time steps to evaluate at.

    Args:
        max_days: Maximum days for daily evaluation range.
        cut_hours: Cutoff for hourly steps; daily steps start after this.
        step_hours: Hourly step size within [0, cut_hours].
        step_days: Daily step size within (cut_hours/24, max_days].

    Returns:
        Sorted list of unique step indices.
    """
    thresholds = []

    for h in range(step_hours, cut_hours + 1, step_hours):
        step = time_to_step(h, 'h')
        if step is not None:
            thresholds.append(step)

    start_day = int(np.ceil(cut_hours / 24))
    for d in range(start_day + 1, max_days + 1, step_days):
        step = time_to_step(d, 'D')
        if step is not None:
            thresholds.append(step)

    return sorted(list(set(thresholds)))


def format_step_label(step):
    """Convert step to human-readable time label."""
    time_min = step_to_time(step)

    if time_min is None:
        return f"Step {step}"

    if time_min < 60:
        return f"{int(time_min)} min"
    elif time_min < 24 * 60:
        hours = time_min / 60
        if hours.is_integer():
            hours = int(hours)
        return f"{hours} h"
    else:
        days = time_min / (24 * 60)
        if days.is_integer():
            days = int(days)
        return f"{days} day" + ("s" if days != 1 else "")


# ============================================================================
# EBM EVALUATION
# ============================================================================

def run_eval_ebm(
    base_df: pd.DataFrame,
    cfg: dict,
    masking_point: pd.Timedelta,
    model,
    model_name: str,
    preprocessor=None,
    categorical_features: Optional[List[str]] = None,
    continuous_features: Optional[List[str]] = None,
    save_predictions: bool = True,
) -> Optional[TimeMetricResult]:
    """
    Evaluate a trained EBM model at a specific masking point.

    Creates AggregatedDS with the given masking_point, applies the model,
    computes AUROC and AUPRC with confidence intervals, and optionally saves
    per-patient predictions to reports/predictions/.

    Args:
        base_df: Base dataframe (test set only).
        cfg: Configuration dictionary.
        masking_point: pd.Timedelta — how much data to reveal.
        model: Trained EBM (ExplainableBoostingClassifier or similar sklearn estimator).
        model_name: Name used for saved artifacts.
        preprocessor: Optional sklearn preprocessor (fit on training data).
            If None, raw features are passed directly to the model.
        categorical_features: List of categorical feature column names.
            Required if preprocessor is provided.
        continuous_features: List of continuous feature column names.
            Required if preprocessor is provided.
        save_predictions: If True, save per-patient predictions CSV.

    Returns:
        TimeMetricResult, or None if evaluation failed (e.g. single class in test set).
    """
    logger.info(f"Evaluating EBM at masking_point={masking_point}")

    # Build aggregated dataset at this masking point
    agg_ds = AggregatedDS(
        cfg=cfg,
        base_df=base_df,
        masking_point=masking_point,
        agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
        concepts=cfg["concepts"],
        default_mode=True,
    )

    X, y = agg_ds.get_X_y()
    y_arr = np.asarray(y).round().astype(int)

    if len(set(y_arr)) < 2:
        logger.warning(f"Skipping masking_point={masking_point}: only one class in test set")
        return None

    # Apply preprocessor if provided
    if preprocessor is not None:
        if categorical_features is None or continuous_features is None:
            raise ValueError(
                "categorical_features and continuous_features must be provided "
                "when a preprocessor is given."
            )
        X_proc = preprocessor.transform(X[categorical_features + continuous_features])
    else:
        X_proc = X

    # Get predictions
    y_proba = model.predict_proba(X_proc)[:, 1]

    # Compute metrics with CIs
    auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(y_arr, y_proba)
    auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(y_arr, y_proba)

    logger.info(f"AUROC={auroc:.3f} [{auroc_lower:.3f}-{auroc_upper:.3f}]  "
                f"AUPRC={auprc:.3f} [{auprc_lower:.3f}-{auprc_upper:.3f}]")

    # Determine censor step from masking_point
    time_min = masking_point.total_seconds() / 60
    censor_step = time_to_step(time_min, 'min')
    if censor_step is None:
        censor_step = -1

    result = TimeMetricResult(
        time_min=time_min,
        time_hours=time_min / 60,
        time_days=time_min / (24 * 60),
        censor_step=censor_step,
        auroc=auroc,
        auroc_ci=(auroc_lower, auroc_upper),
        auprc=auprc,
        auprc_ci=(auprc_lower, auprc_upper),
        n_samples=len(y_arr),
        n_positive=int(y_arr.sum()),
    )

    # Save per-patient predictions
    if save_predictions:
        id_col = cfg["dataset"]["id_col"]
        X_with_id, _ = agg_ds.get_X_y(include_id=True)
        pids = X_with_id[id_col].values

        preds_df = pd.DataFrame({
            id_col: pids,
            "y_true": y_arr,
            "y_pred": y_proba,
            "masking_point": str(masking_point),
            "time_min": time_min,
            "time_hours": result.time_hours,
            "time_days": result.time_days,
        })

        os.makedirs("reports/predictions", exist_ok=True)
        label = format_step_label(censor_step).replace(" ", "")
        save_path = f"reports/predictions/preds_{model_name}_{label}.csv"
        preds_df.to_csv(save_path, index=False)
        logger.info(f"Predictions saved: {save_path}")

    return result


# ============================================================================
# PLOTTING
# ============================================================================

def plot_time_metrics(results: List[TimeMetricResult], cut_hours=72, max_days=30):
    """
    Plot AUROC and AUPRC over time with confidence intervals.

    Args:
        results: List of TimeMetricResult objects.
        cut_hours: Cut-off for the hours subplot.
        max_days: Maximum days for the days subplot.

    Returns:
        matplotlib Figure.
    """
    if not results:
        raise ValueError("No results to plot")

    times_h = np.array([r.time_hours for r in results])
    times_d = np.array([r.time_days for r in results])
    auroc_vals = np.array([r.auroc for r in results])
    auroc_lower = np.array([r.auroc_ci[0] for r in results])
    auroc_upper = np.array([r.auroc_ci[1] for r in results])
    auprc_vals = np.array([r.auprc for r in results])
    auprc_lower = np.array([r.auprc_ci[0] for r in results])
    auprc_upper = np.array([r.auprc_ci[1] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Hours view
    mask_cut = times_h <= cut_hours

    for vals, lower, upper, marker, color, label in [
        (auroc_vals[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut], 'o', "C0", "AUROC"),
        (auprc_vals[mask_cut], auprc_lower[mask_cut], auprc_upper[mask_cut], 's', "C1", "AUPRC"),
    ]:
        x = times_h[mask_cut]
        if len(x) > 0:
            if x[-1] < cut_hours:
                x_ext = np.append(x, cut_hours)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper

            ax1.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax1.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours + 1, 6))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A) Performance over Hours", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.0, 1.0)

    # Plot B: Days view
    for vals, lower, upper, marker, color, label in [
        (auroc_vals, auroc_lower, auroc_upper, 'o', "C0", "AUROC"),
        (auprc_vals, auprc_lower, auprc_upper, 's', "C1", "AUPRC"),
    ]:
        x = times_d
        if len(x) > 0:
            if x[-1] < max_days:
                x_ext = np.append(x, max_days)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper

            ax2.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax2.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days + 1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B) Performance over Days", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    return fig
