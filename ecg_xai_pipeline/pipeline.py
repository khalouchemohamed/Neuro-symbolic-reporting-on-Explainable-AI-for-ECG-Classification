# =============================================================================
# IMPORTS & CONFIGURATION
# =============================================================================
from __future__ import annotations

from datetime import datetime
import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import binary_dilation, gaussian_filter1d
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ecg_xai_pipeline.config import PipelineConfig
from ecg_xai_pipeline.data import (
    apply_clinical_region_occlusion,
    build_occlusion_masks,
    component_duration_rows,
    load_baked_ontology,
    ontology_summary_text,
    prepare_datasets,
    validate_signal,
    zero_tabular_features,
)
from ecg_xai_pipeline.ontology import (
    extract_xai_clinical_features,
    infer_ontology_report,
)
from ecg_xai_pipeline.model import (
    gradcam_attributions,
    integrated_gradients_attributions,
    predict_main,
    select_stratified_samples,
    set_global_seed,
    shap_attributions,
    train_or_load_model,
)


CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Paced/Unknown",
}


# =============================================================================
# UTILITIES & METRICS
# =============================================================================
def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": classification_report(
            y_true, y_pred, target_names=[CLASS_NAMES[i] for i in range(5)]
        ),
    }


def evaluate_clinical_occlusion(
    model,
    bundle,
    config: PipelineConfig,
    baseline_pred: np.ndarray,
) -> dict:
    masks_test = np.asarray(bundle.clinical_masks_test, dtype=np.uint8)
    valid_idx = np.where(masks_test.sum(axis=1) > 0)[0]
    if len(valid_idx) == 0:
        return {
            "occlusion_valid_samples": 0,
            "baseline_accuracy": 0.0,
            "occluded_accuracy": 0.0,
            "accuracy_drop": 0.0,
            "baseline_macro_f1": 0.0,
            "occluded_macro_f1": 0.0,
            "macro_f1_drop": 0.0,
            "baseline_balanced_accuracy": 0.0,
            "occluded_balanced_accuracy": 0.0,
            "balanced_accuracy_drop": 0.0,
            "mean_mask_coverage": 0.0,
            "flipped_correct_to_wrong": 0,
            "per_sample_rows": [],
            "interpretation": "No test samples with non-empty clinical masks.",
        }

    X_valid = bundle.X_test[valid_idx]
    F_valid = bundle.F_test[valid_idx]
    y_valid = bundle.y_test[valid_idx]
    masks_valid = masks_test[valid_idx]

    baseline_pred_valid = baseline_pred[valid_idx]
    occlusion_masks = build_occlusion_masks(
        masks_valid, bundle.X_test_raw[valid_idx], config
    )
    mean_mask_coverage = float(occlusion_masks.mean())

    X_occluded = apply_clinical_region_occlusion(X_valid, occlusion_masks)
    if config.clinical_occlusion_zero_features:
        F_occluded = zero_tabular_features(F_valid)
    else:
        F_occluded = F_valid.copy()

    _, occluded_pred = predict_main(
        model, X_occluded, F_occluded, batch_size=config.batch_size
    )

    baseline_summary = classification_summary(y_valid, baseline_pred_valid)
    occluded_summary = classification_summary(y_valid, occluded_pred)

    correct_baseline = baseline_pred_valid == y_valid
    wrong_occluded = occluded_pred != y_valid
    flipped = int(np.logical_and(correct_baseline, wrong_occluded).sum())

    per_sample_rows = []
    for local_idx, sample_idx in enumerate(valid_idx):
        per_sample_rows.append(
            {
                "sample_idx": int(sample_idx),
                "true_label": int(y_valid[local_idx]),
                "baseline_pred": int(baseline_pred_valid[local_idx]),
                "occluded_pred": int(occluded_pred[local_idx]),
                "flipped_correct_to_wrong": bool(
                    correct_baseline[local_idx] and wrong_occluded[local_idx]
                ),
            }
        )

    accuracy_drop = baseline_summary["accuracy"] - occluded_summary["accuracy"]
    macro_f1_drop = baseline_summary["macro_f1"] - occluded_summary["macro_f1"]
    balanced_accuracy_drop = (
        baseline_summary["balanced_accuracy"] - occluded_summary["balanced_accuracy"]
    )

    if accuracy_drop >= 0.25 or balanced_accuracy_drop >= 0.40:
        interpretation = (
            "Large accuracy and balanced-accuracy drop after occluding the active "
            "beat envelope confirms the model depends on clinically important "
            "waveform regions rather than residual baseline cues."
        )
    elif accuracy_drop >= 0.10 or balanced_accuracy_drop >= 0.20:
        interpretation = (
            "Moderate drop suggests partial dependence on clinical regions; "
            "residual beat transitions may still carry class information."
        )
    else:
        interpretation = (
            "Small drop suggests the model may still exploit non-clinical cues "
            "outside the occluded envelope."
        )

    return {
        "occlusion_valid_samples": int(len(valid_idx)),
        "occlusion_mode": config.clinical_occlusion_mode,
        "occlusion_dilation_radius": config.clinical_occlusion_dilation_radius,
        "mean_mask_coverage": mean_mask_coverage,
        "baseline_accuracy": baseline_summary["accuracy"],
        "occluded_accuracy": occluded_summary["accuracy"],
        "accuracy_drop": float(accuracy_drop),
        "baseline_macro_f1": baseline_summary["macro_f1"],
        "occluded_macro_f1": occluded_summary["macro_f1"],
        "macro_f1_drop": float(macro_f1_drop),
        "baseline_balanced_accuracy": baseline_summary["balanced_accuracy"],
        "occluded_balanced_accuracy": occluded_summary["balanced_accuracy"],
        "balanced_accuracy_drop": float(balanced_accuracy_drop),
        "flipped_correct_to_wrong": flipped,
        "per_sample_rows": per_sample_rows,
        "interpretation": interpretation,
        "occluded_report": occluded_summary["report"],
    }


def occlusion_telemetry_lines(occlusion_result: dict) -> list[str]:
    return [
        f"occlusion_valid_samples={occlusion_result['occlusion_valid_samples']}",
        f"baseline_accuracy={occlusion_result['baseline_accuracy']:.4f}",
        f"occluded_accuracy={occlusion_result['occluded_accuracy']:.4f}",
        f"accuracy_drop={occlusion_result['accuracy_drop']:.4f}",
        f"baseline_macro_f1={occlusion_result['baseline_macro_f1']:.4f}",
        f"occluded_macro_f1={occlusion_result['occluded_macro_f1']:.4f}",
        f"macro_f1_drop={occlusion_result['macro_f1_drop']:.4f}",
        f"baseline_balanced_accuracy={occlusion_result['baseline_balanced_accuracy']:.4f}",
        f"occluded_balanced_accuracy={occlusion_result['occluded_balanced_accuracy']:.4f}",
        f"balanced_accuracy_drop={occlusion_result['balanced_accuracy_drop']:.4f}",
        f"occlusion_mode={occlusion_result['occlusion_mode']}",
        f"occlusion_dilation_radius={occlusion_result['occlusion_dilation_radius']}",
        f"mean_mask_coverage={occlusion_result['mean_mask_coverage']:.4f}",
        f"flipped_correct_to_wrong={occlusion_result['flipped_correct_to_wrong']}",
    ]


def write_clinical_occlusion_reports(
    occlusion_result: dict, config: PipelineConfig
) -> None:
    summary_lines = [
        "Clinical Region Occlusion Test (inference-time)",
        "=============================================",
        f"Occlusion mode: {occlusion_result['occlusion_mode']}",
        f"Dilation radius: {occlusion_result['occlusion_dilation_radius']} samples",
        f"Mean masked coverage: {occlusion_result['mean_mask_coverage']:.1%}",
        f"Valid test samples: {occlusion_result['occlusion_valid_samples']}",
        f"Baseline accuracy: {occlusion_result['baseline_accuracy']:.4f}",
        f"Occluded accuracy: {occlusion_result['occluded_accuracy']:.4f}",
        f"Accuracy drop: {occlusion_result['accuracy_drop']:.4f}",
        f"Baseline balanced accuracy: {occlusion_result['baseline_balanced_accuracy']:.4f}",
        f"Occluded balanced accuracy: {occlusion_result['occluded_balanced_accuracy']:.4f}",
        f"Balanced accuracy drop: {occlusion_result['balanced_accuracy_drop']:.4f}",
        f"Baseline macro F1: {occlusion_result['baseline_macro_f1']:.4f}",
        f"Occluded macro F1: {occlusion_result['occluded_macro_f1']:.4f}",
        f"Macro F1 drop: {occlusion_result['macro_f1_drop']:.4f}",
        f"Flipped correct->wrong: {occlusion_result['flipped_correct_to_wrong']}",
        "",
        "Interpretation:",
        occlusion_result["interpretation"],
    ]
    if "occluded_report" in occlusion_result:
        summary_lines.extend(
            ["", "Occluded classification report:", occlusion_result["occluded_report"]]
        )

    _write_text(
        config.reports_dir / "clinical_occlusion_summary.txt",
        "\n".join(summary_lines),
    )
    save_dataframe(
        occlusion_result["per_sample_rows"],
        config.reports_dir / "clinical_occlusion_per_sample.csv",
    )
    plot_clinical_occlusion_comparison(
        occlusion_result,
        config.figures_dir / "clinical_occlusion_comparison.png",
    )


def attribution_to_importance(attribution: np.ndarray, sigma: float) -> np.ndarray:
    values = np.abs(np.asarray(attribution, dtype=np.float32).reshape(-1))
    if sigma > 0:
        values = gaussian_filter1d(values, sigma=sigma)
    max_value = float(values.max()) if values.size else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_value).astype(np.float32)


def suppress_edge_importance(importance: np.ndarray, ignore_edge: int) -> np.ndarray:
    values = np.asarray(importance, dtype=np.float32).reshape(-1).copy()
    if ignore_edge <= 0:
        return values
    edge = min(ignore_edge, len(values) // 2)
    values[:edge] = 0.0
    values[-edge:] = 0.0
    return values


def _mask_from_threshold(
    importance: np.ndarray, threshold: float, dilation_radius: int
) -> np.ndarray:
    mask = (importance >= threshold).astype(np.uint8)
    if dilation_radius > 0:
        structure = np.ones(dilation_radius * 2 + 1, dtype=bool)
        mask = binary_dilation(mask, structure=structure).astype(np.uint8)
    return mask


def importance_to_binary_mask(
    importance: np.ndarray,
    percentile: float,
    dilation_radius: int,
    target_coverage: float | None = None,
) -> tuple[np.ndarray, float]:
    importance = np.asarray(importance, dtype=np.float32).reshape(-1)
    nonzero = importance[importance > 0]
    if len(nonzero) == 0:
        return np.zeros_like(importance, dtype=np.uint8), 0.0

    if target_coverage is not None:
        target = float(np.clip(target_coverage, 0.06, 0.45))
        lo, hi = 0.0, 100.0
        best_mask = None
        best_threshold = float(np.percentile(nonzero, percentile))
        best_gap = float("inf")

        for _ in range(14):
            mid = (lo + hi) / 2.0
            threshold = float(np.percentile(nonzero, mid))
            mask = _mask_from_threshold(importance, threshold, dilation_radius)
            coverage = float(np.mean(mask > 0))
            gap = abs(coverage - target)
            if gap < best_gap:
                best_gap = gap
                best_mask = mask
                best_threshold = threshold
            if coverage > target:
                lo = mid
            else:
                hi = mid

        return best_mask.astype(np.uint8), best_threshold

    threshold = float(np.percentile(nonzero, percentile))
    mask = _mask_from_threshold(importance, threshold, dilation_radius)
    return mask, threshold


def mask_overlap_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return a normalized overlap score for two binary masks.

    This is the F1-style overlap formula used for Dice-style mask agreement.
    """
    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def focused_similarity(
    xai_mask: np.ndarray, medical_mask: np.ndarray, ignore_edge: int
) -> tuple[float, float, float]:
    xai_mask = np.asarray(xai_mask, dtype=bool).reshape(-1)
    medical_mask = np.asarray(medical_mask, dtype=bool).reshape(-1)
    n = len(xai_mask)
    valid = np.zeros(n, dtype=bool)
    start = min(ignore_edge, n // 2)
    end = max(n - ignore_edge, n // 2)
    valid[start:end] = True

    xai_valid = xai_mask & valid
    medical_valid = medical_mask & valid
    dice = mask_overlap_score(xai_valid, medical_valid)
    edge_focus = float((xai_mask & ~valid).sum() / max(xai_mask.sum(), 1))
    alignment_score = dice * (1.0 - edge_focus)
    return dice, alignment_score, edge_focus


def inter_method_agreement(mask_by_method: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for method_a, method_b in itertools.combinations(sorted(mask_by_method), 2):
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "agreement_score": mask_overlap_score(
                    mask_by_method[method_a], mask_by_method[method_b]
                ),
            }
        )
    return rows


def save_dataframe(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# =============================================================================
# VISUALIZATION & PLOTTING
# =============================================================================
def _configure_matplotlib() -> None:
    """Apply publication-quality rcParams.  Called once before first plot."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


_MPL_CONFIGURED = False


def _ensure_matplotlib() -> None:
    """Lazy one-shot matplotlib configuration."""
    global _MPL_CONFIGURED
    if not _MPL_CONFIGURED:
        _configure_matplotlib()
        _MPL_CONFIGURED = True


def _percentage_label(value: float) -> str:
    return f"{value * 100:.1f}%"


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            _percentage_label(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path) -> None:
    _ensure_matplotlib()
    labels = [CLASS_NAMES[i] for i in range(5)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    row_totals = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )
    annotations = np.empty_like(cm, dtype=object)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            annotations[row_idx, col_idx] = (
                f"{cm[row_idx, col_idx]}\n{percentages[row_idx, col_idx] * 100:.1f}%"
            )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("MIT-BIH ECG Classification Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_clinical_occlusion_comparison(occlusion_result: dict, save_path) -> None:
    _ensure_matplotlib()
    metrics = [
        ("Accuracy", occlusion_result["baseline_accuracy"], occlusion_result["occluded_accuracy"]),
        ("Macro F1", occlusion_result["baseline_macro_f1"], occlusion_result["occluded_macro_f1"]),
    ]
    labels = [name for name, _, _ in metrics]
    baseline_values = [base for _, base, _ in metrics]
    occluded_values = [occ for _, _, occ in metrics]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="baseline model",
        color="#1f77b4",
    )
    occluded_bars = ax.bar(
        x + width / 2,
        occluded_values,
        width,
        label="active-beat regions occluded",
        color="#d62728",
    )
    _annotate_bars(ax, baseline_bars)
    _annotate_bars(ax, occluded_bars)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Clinical Region Occlusion vs Baseline (valid test samples)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        framealpha=0.35,
        facecolor="white",
        edgecolor="none",
    )
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _history_series(history: dict[str, list[float]], preferred: list[str]) -> tuple[str, list[float]] | tuple[None, None]:
    for key in preferred:
        values = history.get(key)
        if values:
            return key, list(values)
    return None, None


def save_training_history(history: dict[str, list[float]] | None, config: PipelineConfig) -> None:
    if not history:
        return

    history_df = pd.DataFrame(history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(config.reports_dir / "training_history.csv", index=False)

    plot_training_curve(
        history,
        metric_key_candidates=["loss", "main_output_loss"],
        validation_key_candidates=["val_loss", "val_main_output_loss"],
        ylabel="Loss",
        title="Training Loss by Epoch",
        save_path=config.figures_dir / "training_loss.png",
        as_percentage=False,
    )
    plot_training_curve(
        history,
        metric_key_candidates=["main_output_accuracy", "accuracy"],
        validation_key_candidates=["val_main_output_accuracy", "val_accuracy"],
        ylabel="Accuracy",
        title="Training Accuracy by Epoch",
        save_path=config.figures_dir / "training_accuracy.png",
        as_percentage=True,
    )


def plot_training_curve(
    history: dict[str, list[float]],
    metric_key_candidates: list[str],
    validation_key_candidates: list[str],
    ylabel: str,
    title: str,
    save_path,
    as_percentage: bool,
) -> None:
    _ensure_matplotlib()
    metric_key, train_values = _history_series(history, metric_key_candidates)
    val_key, val_values = _history_series(history, validation_key_candidates)
    if not train_values and not val_values:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train_values:
        y = np.asarray(train_values, dtype=float)
        if as_percentage:
            y = y * 100.0
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", linewidth=2, label="Training")
    if val_values:
        y = np.asarray(val_values, dtype=float)
        if as_percentage:
            y = y * 100.0
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", linewidth=2, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{ylabel} (%)" if as_percentage else ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_xai_overlay(
    signal: np.ndarray,
    importance: np.ndarray,
    medical_mask: np.ndarray,
    method: str,
    sample_idx: int,
    y_true: int,
    y_pred: int,
    save_path,
) -> None:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    importance = np.asarray(importance, dtype=np.float32).reshape(-1)
    medical_mask = np.asarray(medical_mask, dtype=np.uint8).reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(signal, color="black", linewidth=1.3, label="ECG")
    axes[0].fill_between(
        np.arange(len(signal)),
        signal.min(),
        signal.max(),
        where=medical_mask > 0,
        color="#2ca02c",
        alpha=0.20,
        label="Clinical mask",
    )
    for idx in range(len(signal) - 1):
        if importance[idx] > 0:
            axes[0].axvspan(
                idx,
                idx + 1,
                color="#d62728",
                alpha=float(min(0.65 * importance[idx], 0.65)),
            )
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"{method} | sample {sample_idx} | true {int(y_true)} -> pred {int(y_pred)}"
    )

    axes[1].plot(importance, color="#d62728", linewidth=1.5, label=f"{method} importance")
    axes[1].fill_between(
        np.arange(len(medical_mask)),
        0,
        medical_mask,
        step="pre",
        color="#2ca02c",
        alpha=0.30,
        label="Clinical mask",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Sample index within cropped beat")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_xai_method_comparison(
    signal: np.ndarray,
    importance_by_method: dict[str, np.ndarray],
    medical_mask: np.ndarray,
    sample_idx: int,
    y_true: int,
    y_pred: int,
    save_path,
) -> None:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    medical_mask = np.asarray(medical_mask, dtype=np.uint8).reshape(-1)
    methods = list(importance_by_method)
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 2.7 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        importance = np.asarray(importance_by_method[method], dtype=np.float32).reshape(-1)
        ax.plot(signal, color="black", linewidth=1.1, label="ECG")
        ax.fill_between(
            np.arange(len(signal)),
            signal.min(),
            signal.max(),
            where=medical_mask > 0,
            color="#2ca02c",
            alpha=0.18,
            label="Clinical mask",
        )
        scaled = importance * max(float(signal.max() - signal.min()), 1e-6) + float(signal.min())
        ax.plot(scaled, color="#d62728", linewidth=1.4, label=f"{method} importance")
        ax.set_ylabel(method)
        ax.legend(loc="upper right")

    axes[0].set_title(
        f"XAI method comparison | sample {sample_idx} | true {int(y_true)} -> pred {int(y_pred)}"
    )
    axes[-1].set_xlabel("Sample index within cropped beat")
    plt.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_xai_metric_summary(xai_df, save_path) -> None:
    _ensure_matplotlib()
    valid = xai_df[xai_df["comparison_valid"].astype(bool)].copy()
    if valid.empty:
        return
    if "dice" not in valid.columns and "alignment_score" in valid.columns:
        valid["dice"] = valid["alignment_score"]

    metrics = ["dice", "alignment_score", "edge_focus"]
    summary = valid.groupby("method")[metrics].median().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, metric, color in zip(axes, metrics, ["#1f77b4", "#2ca02c", "#d62728"]):
        sns.barplot(data=summary, x="method", y=metric, ax=ax, color=color)
        _annotate_bars(ax, ax.patches)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Median XAI Clinical Alignment by Method")
    plt.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_inter_method_agreement(agreement_df, save_path) -> None:
    if agreement_df.empty:
        return
    methods = sorted(set(agreement_df["method_a"]).union(set(agreement_df["method_b"])))
    matrix = np.eye(len(methods), dtype=np.float32)
    index = {method: idx for idx, method in enumerate(methods)}
    for _, row in agreement_df.iterrows():
        i = index[row["method_a"]]
        j = index[row["method_b"]]
        matrix[i, j] = matrix[j, i] = float(row["agreement_score"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=methods,
        yticklabels=methods,
        cbar=True,
    )
    plt.title("Mean Inter-Method XAI Agreement")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()


def _comparison_is_valid(validation: dict) -> bool:
    if not validation.get("success", False):
        return False
    if np.asarray(validation.get("mask", [])).sum() == 0:
        return False
    return len(validation.get("intervals", {}).get("qrs", [])) > 0


def _write_text(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _alignment_grade(score: float) -> str:
    if np.isnan(score):
        return "not_available"
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "minimal"


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
def run_pipeline(config: PipelineConfig) -> dict:
    start_time = datetime.now()
    config.ensure_dirs()
    set_global_seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)

    print("[1/8] Loading baked-in ECG ontology...")
    ontology_graph, ontology_status = load_baked_ontology()
    _write_text(
        config.reports_dir / "ontology_status.txt",
        ontology_summary_text(ontology_status),
    )
    ontology_rows = component_duration_rows(ontology_graph)
    if ontology_rows:
        save_dataframe(ontology_rows, config.reports_dir / "ontology_component_durations.csv")
    print(ontology_status.message)

    print("[2/8] Preparing Kaggle MIT-BIH CSV data, clinical masks, and tabular features...")
    bundle = prepare_datasets(config)

    print("[3/8] Training or loading TCN + CBAM hybrid model...")
    model, training_history = train_or_load_model(bundle, config)
    save_training_history(training_history, config)

    print("[4/8] Evaluating test-set performance...")
    main_probs, y_pred = predict_main(
        model, bundle.X_test, bundle.F_test, batch_size=config.batch_size
    )
    summary = classification_summary(bundle.y_test, y_pred)
    plot_confusion_matrix(
        bundle.y_test, y_pred, config.figures_dir / "confusion_matrix.png"
    )
    _write_text(config.reports_dir / "classification_report.txt", summary["report"])

    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")

    occlusion_result = None
    if config.run_clinical_occlusion_test:
        print("[5/8] Running clinical region occlusion test...")
        occlusion_result = evaluate_clinical_occlusion(
            model=model,
            bundle=bundle,
            config=config,
            baseline_pred=y_pred,
        )
        write_clinical_occlusion_reports(occlusion_result, config)
        print(
            f"       -> Valid samples: {occlusion_result['occlusion_valid_samples']}, "
            f"accuracy drop: {occlusion_result['accuracy_drop']:.4f}"
        )
        print(f"       -> {occlusion_result['interpretation']}")

    if not config.run_xai:
        telemetry_lines = [
            f"accuracy={summary['accuracy']:.4f}",
            f"macro_f1={summary['macro_f1']:.4f}",
        ]
        if occlusion_result is not None:
            telemetry_lines.extend(occlusion_telemetry_lines(occlusion_result))
        telemetry_lines.append(f"elapsed={datetime.now() - start_time}")
        _write_text(
            config.reports_dir / "experiment_summary.txt",
            "\n".join(telemetry_lines),
        )
        return {
            "summary": summary,
            "occlusion": occlusion_result,
            "elapsed": datetime.now() - start_time,
        }

    print("[6/8] Computing SHAP, GradCAM, and Integrated Gradients...")
    sample_indices = select_stratified_samples(bundle.y_test, config, rng)
    print(f"       -> {len(sample_indices)} stratified samples selected")
    bg_size = min(config.shap_background_size, len(bundle.X_train))
    bg_idx = rng.choice(len(bundle.X_train), size=bg_size, replace=False)

    print("       -> Running SHAP (GradientExplainer)...")
    shap_result = shap_attributions(
        model=model,
        X_background=bundle.X_train[bg_idx],
        F_background=bundle.F_train_model[bg_idx],
        X_samples=bundle.X_test[sample_indices],
        F_samples=bundle.F_test[sample_indices],
        predicted_classes=y_pred[sample_indices],
        use_aux_head=config.shap_use_aux_head,
    )
    print("       -> Running GradCAM...")
    gradcam_result = gradcam_attributions(
        model=model,
        X_samples=bundle.X_test[sample_indices],
        F_samples=bundle.F_test[sample_indices],
        predicted_classes=y_pred[sample_indices],
        layer_name=config.gradcam_layer_name,
        use_aux_head=config.gradcam_use_aux_head,
        score_mode=config.xai_score_mode,
    )
    print("       -> Running Integrated Gradients "
          f"({config.integrated_gradients_steps} steps x "
          f"{config.integrated_gradients_smooth_samples} smooth)...")
    ig_result = integrated_gradients_attributions(
        model=model,
        X_samples=bundle.X_test[sample_indices],
        F_samples=bundle.F_test[sample_indices],
        predicted_classes=y_pred[sample_indices],
        config=config,
        X_background=bundle.X_train[bg_idx],
    )
    attributions = {
        "SHAP": shap_result,
        "GradCAM": gradcam_result,
        "IntegratedGradients": ig_result,
    }
    print("       -> All XAI attributions computed.")

    print("[7/8] Comparing XAI masks against clinical morphology masks...")
    rows = []
    agreement_rows = []
    fallback_count = 0

    for local_idx, sample_idx in enumerate(sample_indices):
        validation_bundle = validate_signal(
            raw_signal=bundle.X_test_raw[sample_idx],
            sample_idx=sample_idx,
            label=int(y_pred[sample_idx]),
            config=config,
        )
        validation = validation_bundle["validation"]
        medical_mask = np.asarray(validation_bundle["mask"], dtype=np.uint8)
        if config.xai_zero_edges_before_threshold:
            medical_mask = suppress_edge_importance(medical_mask, config.ignore_edge).astype(np.uint8)
        comparison_valid = _comparison_is_valid(validation) and medical_mask.sum() > 0
        if not comparison_valid:
            fallback_count += 1
            
        masks_for_agreement = {}
        importance_for_comparison = {}
        target_coverage = None
        if config.xai_match_clinical_coverage and comparison_valid:
            target_coverage = float(np.mean(medical_mask > 0))
        threshold_policy = (
            "clinical_coverage" if target_coverage is not None else "fixed_percentile"
        )

        for method, values in attributions.items():
            importance = attribution_to_importance(
                values[local_idx], sigma=config.xai_smoothing_sigma
            )
            if config.xai_zero_edges_before_threshold:
                importance = suppress_edge_importance(importance, config.ignore_edge)
            importance_for_comparison[method] = importance
            xai_mask, threshold = importance_to_binary_mask(
                importance,
                percentile=config.xai_percentile,
                dilation_radius=config.dilation_radius,
                target_coverage=target_coverage,
            )
            masks_for_agreement[method] = xai_mask

            if comparison_valid:
                dice, alignment_score, edge_focus = focused_similarity(
                    xai_mask, medical_mask, ignore_edge=config.ignore_edge
                )
            else:
                dice, alignment_score, edge_focus = np.nan, np.nan, np.nan

            plot_xai_overlay(
                signal=bundle.X_test_raw[sample_idx],
                importance=importance,
                medical_mask=medical_mask,
                method=method,
                sample_idx=sample_idx,
                y_true=int(bundle.y_test[sample_idx]),
                y_pred=int(y_pred[sample_idx]),
                save_path=config.xai_dir / f"{method.lower()}_sample_{sample_idx}.png",
            )

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "method": method,
                    "y_true": int(bundle.y_test[sample_idx]),
                    "y_pred": int(y_pred[sample_idx]),
                    "pred_confidence": float(main_probs[sample_idx, y_pred[sample_idx]]),
                    "comparison_valid": bool(comparison_valid),
                    "dice": dice,
                    "alignment_score": alignment_score,
                    "alignment_grade": _alignment_grade(float(alignment_score)),
                    "edge_focus": edge_focus,
                    "threshold": float(threshold),
                    "threshold_policy": threshold_policy,
                    "xai_coverage": float(np.mean(xai_mask > 0)),
                    "clinical_coverage": float(np.mean(medical_mask > 0)),
                    "coverage_delta": abs(
                        float(np.mean(xai_mask > 0)) - float(np.mean(medical_mask > 0))
                    ),
                }
            )

        plot_xai_method_comparison(
            signal=bundle.X_test_raw[sample_idx],
            importance_by_method=importance_for_comparison,
            medical_mask=medical_mask,
            sample_idx=sample_idx,
            y_true=int(bundle.y_test[sample_idx]),
            y_pred=int(y_pred[sample_idx]),
            save_path=config.xai_dir / f"method_comparison_sample_{sample_idx}.png",
        )

        xai_features = extract_xai_clinical_features(
            validation_bundle, importance_for_comparison,
            bundle.X_test_raw[sample_idx], config,
        )
        report_text = infer_ontology_report(
            xai_features, int(y_pred[sample_idx]), ontology_graph,
        )
        _write_text(
            config.xai_dir / f"rapport_sample_{sample_idx}.txt", report_text,
        )

        for agreement in inter_method_agreement(masks_for_agreement):
            agreement["sample_idx"] = int(sample_idx)
            agreement_rows.append(agreement)

    total_xai_samples = len(sample_indices)
    fallback_percentage = (fallback_count / total_xai_samples) * 100.0 if total_xai_samples > 0 else 0.0
    print(f"       -> Reference mask fallback rate: {fallback_count}/{total_xai_samples} ({fallback_percentage:.1f}%)")

    xai_df = save_dataframe(rows, config.reports_dir / "xai_clinical_comparison.csv")
    agreement_df = save_dataframe(
        agreement_rows, config.reports_dir / "xai_inter_method_agreement.csv"
    )
    plot_xai_metric_summary(xai_df, config.figures_dir / "xai_method_metric_summary.png")
    if not agreement_df.empty:
        mean_agreement = (
            agreement_df.groupby(["method_a", "method_b"], as_index=False)["agreement_score"]
            .mean()
        )
        plot_inter_method_agreement(
            mean_agreement, config.figures_dir / "xai_inter_method_agreement.png"
        )

    print("[8/8] Writing experiment summary...")
    valid = xai_df[xai_df["comparison_valid"].astype(bool)]
    telemetry_lines = [
        f"accuracy={summary['accuracy']:.4f}",
        f"macro_f1={summary['macro_f1']:.4f}",
        f"xai_samples={len(sample_indices)}",
        f"xai_fallback_count={fallback_count}",
        f"xai_fallback_rate={fallback_percentage:.1f}%",
    ]
    if not valid.empty:
        telemetry_lines.extend(
            [
                f"median_dice={valid['dice'].median():.4f}",
                f"mean_dice={valid['dice'].mean():.4f}",
                f"median_alignment_score={valid['alignment_score'].median():.4f}",
                f"mean_alignment_score={valid['alignment_score'].mean():.4f}",
                f"mean_edge_focus={valid['edge_focus'].mean():.4f}",
                f"mean_coverage_delta={valid['coverage_delta'].mean():.4f}",
            ]
        )
    if not agreement_df.empty:
        telemetry_lines.append(
            f"mean_inter_method_agreement={agreement_df['agreement_score'].mean():.4f}"
        )
    if occlusion_result is not None:
        telemetry_lines.extend(occlusion_telemetry_lines(occlusion_result))

    if not valid.empty:
        print("      XAI clinical alignment summary:")
        method_summary = valid.groupby("method")[["dice", "alignment_score"]].median()
        for method, row in method_summary.sort_index().iterrows():
            print(
                f"        {method:20s} -> Median Dice: {row['dice']:.4f}, "
                f"Median alignment: {row['alignment_score']:.4f}"
            )

    telemetry_lines.append(f"elapsed={datetime.now() - start_time}")
    _write_text(config.reports_dir / "experiment_summary.txt", "\n".join(telemetry_lines))

    return {
        "summary": summary,
        "occlusion": occlusion_result,
        "xai": xai_df,
        "agreement": agreement_df,
        "elapsed": datetime.now() - start_time,
    }
