# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from sklearn.model_selection import train_test_split

from ecg_xai_pipeline.config import PipelineConfig


# =============================================================================
# ONTOLOGY (re-exported from ecg_xai_pipeline.ontology)
# =============================================================================
from ecg_xai_pipeline.ontology import (  # noqa: F401
    BAKED_IN_ONTOLOGY_TTL,
    OntologyStatus,
    component_duration_rows,
    load_baked_ontology,
    ontology_summary_text,
)


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class DatasetBundle:
    X_train: np.ndarray
    F_train: np.ndarray
    y_train: np.ndarray
    X_train_clean: np.ndarray
    F_train_clean: np.ndarray
    F_train_model: np.ndarray
    X_val: np.ndarray
    F_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    F_test: np.ndarray
    y_test: np.ndarray
    X_test_raw: np.ndarray
    clinical_masks_train: np.ndarray | None = None
    clinical_masks_val: np.ndarray | None = None


# =============================================================================
# DATA VALIDATION
# =============================================================================
def validate_mitbih_frame(df: pd.DataFrame, split_name: str, config: PipelineConfig) -> None:
    min_columns = config.crop_end + 1
    if df.shape[1] < min_columns:
        raise ValueError(
            f"{split_name} has {df.shape[1]} columns, but crop_end={config.crop_end} "
            f"requires at least {min_columns} columns including the label."
        )

    labels = df.iloc[:, -1].to_numpy()
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"{split_name} contains non-finite labels.")
    if not np.all(np.equal(labels, labels.astype(np.int64))):
        raise ValueError(f"{split_name} labels must be integer class ids.")

    bad_labels = sorted(set(labels.astype(np.int64)) - set(range(config.num_classes)))
    if bad_labels:
        raise ValueError(
            f"{split_name} contains labels outside 0..{config.num_classes - 1}: "
            f"{bad_labels}"
        )

    signal_values = df.iloc[:, :-1].to_numpy(dtype=np.float32, copy=False)
    if not np.all(np.isfinite(signal_values)):
        raise ValueError(f"{split_name} contains non-finite signal values.")


# =============================================================================
# SIGNAL PROCESSING (inline — no external dependencies)
# =============================================================================
_CLASS_PROFILES = {
    0: {"name": "Normal Beat",
        "p_present": True, "t_present": True,
        "qrs_ms": (80, 110), "p_width_ms": (80, 120)},
    1: {"name": "Supraventricular Ectopic Beat",
        "p_present": True, "t_present": True,
        "qrs_ms": (70, 110), "p_width_ms": (70, 120)},
    2: {"name": "Ventricular Ectopic Beat",
        "p_present": False, "t_present": True,
        "qrs_ms": (120, 200), "p_width_ms": None},
    3: {"name": "Fusion Beat",
        "p_present": True, "t_present": True,
        "qrs_ms": (100, 160), "p_width_ms": (80, 130)},
    4: {"name": "Paced / Unknown Beat",
        "p_present": False, "t_present": True,
        "qrs_ms": (80, 200), "p_width_ms": None},
}


def _get_profile(rhythm_label):
    if rhythm_label is None:
        return _CLASS_PROFILES[0]
    return _CLASS_PROFILES.get(int(rhythm_label), _CLASS_PROFILES[0])


def _ms_to_samples(ms, sampling_rate):
    return max(1, int(round((ms / 1000.0) * sampling_rate)))


def _safe_slice(start, end, signal_length):
    start = max(0, int(start or 0))
    end = min(signal_length, int(end or signal_length))
    return max(0, start), max(start, end)


def _mark_segment(mask, start, end):
    start, end = _safe_slice(start, end, len(mask))
    if end > start:
        mask[start:end] = 1


def _find_active_end(signal, gradient_floor=0.01, min_tail=8):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    grad = np.abs(np.diff(signal, prepend=signal[0]))
    near_zero = (np.abs(signal) < 0.03) & (grad < gradient_floor)
    # Find the longest trailing run of near-zero samples
    if not near_zero[-1]:
        return len(signal)
    # Reverse scan: find the first non-near-zero from the end
    rev = near_zero[::-1]
    non_zero_positions = np.nonzero(~rev)[0]
    if len(non_zero_positions) == 0:
        return 0  # entire signal is near-zero
    run_length = int(non_zero_positions[0])
    if run_length >= min_tail:
        return len(signal) - run_length
    return len(signal)


def _find_dominant_qrs_peak(signal, active_end):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if active_end <= 0:
        return None
    left_margin = min(max(8, int(round(active_end * 0.06))), max(active_end - 3, 0))
    right_margin = min(max(4, int(round(active_end * 0.03))), max(active_end - 1, 0))
    search_start = left_margin
    search_end = max(search_start + 1, active_end - right_margin)
    if search_end - search_start < 3:
        search_start, search_end = 0, active_end

    seg = signal[search_start:search_end]
    n = len(seg)
    if n < 3:
        return None

    # Vectorized local-maximum detection
    amp = np.abs(seg)
    is_peak = np.zeros(n, dtype=bool)
    is_peak[1:-1] = (amp[1:-1] >= amp[:-2]) & (amp[1:-1] >= amp[2:])

    # Dynamic amplitude threshold based on overall signal scale
    max_amp = float(np.max(np.abs(signal)))
    qrs_threshold = max(0.05, min(0.15, max_amp * 0.25))

    peak_indices = np.nonzero(is_peak)[0]
    if len(peak_indices) == 0:
        best_local = int(np.argmax(amp))
        best_idx = search_start + best_local
        return int(best_idx) if abs(signal[best_idx]) >= qrs_threshold else None

    # Vectorized sharpness + center-bias scoring
    sharpness = np.abs(seg[peak_indices] - seg[peak_indices - 1]) + \
                np.abs(seg[peak_indices] - seg[peak_indices + 1])
    window_center = n / 2.0
    window_span = max(n, 1)
    center_bias = np.maximum(
        1.0 - 0.35 * (np.abs(peak_indices - window_center) / window_span), 0.4
    )
    scores = amp[peak_indices] * (1.0 + sharpness) * center_bias

    best_local = peak_indices[int(np.argmax(scores))]
    best_idx = search_start + int(best_local)
    return int(best_idx) if abs(signal[best_idx]) >= qrs_threshold else None


def _build_qrs_interval_from_peak(peak_idx, signal_length, sampling_rate, rhythm_label):
    profile = _get_profile(rhythm_label)
    qrs_width = _ms_to_samples(int(round(np.mean(profile["qrs_ms"]))), sampling_rate)
    half_width = max(6, qrs_width // 2)
    return _safe_slice(peak_idx - half_width, peak_idx + half_width + 1, signal_length)


def _refine_interval_around_peak(signal, peak_idx, left_limit, right_limit, max_width, ratio=0.2):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    peak_idx, left_limit = int(peak_idx), max(0, int(left_limit))
    right_limit = min(len(signal), int(right_limit))
    if peak_idx < left_limit or peak_idx >= right_limit:
        return _safe_slice(peak_idx, peak_idx + 1, len(signal))
    local_start = max(left_limit, peak_idx - max_width)
    local_end = min(right_limit, peak_idx + max_width + 1)
    baseline = float(np.median(signal[local_start:local_end]))
    threshold = max(0.02, abs(signal[peak_idx] - baseline) * ratio)

    dev = np.abs(signal - baseline)
    below = dev < threshold

    # Left boundary: find rightmost index < peak_idx where two consecutive below-threshold
    left_region = below[left_limit:peak_idx]
    if len(left_region) >= 2:
        both_below = left_region[:-1] & left_region[1:]
        crossings = np.nonzero(both_below)[0]
        left = (left_limit + int(crossings[-1]) + 1) if len(crossings) > 0 else left_limit
    else:
        left = left_limit

    # Right boundary: find leftmost index > peak_idx where two consecutive below-threshold
    right_region = below[peak_idx:right_limit]
    if len(right_region) >= 2:
        both_below = right_region[:-1] & right_region[1:]
        crossings = np.nonzero(both_below)[0]
        right = (peak_idx + int(crossings[0]) + 1) if len(crossings) > 0 else right_limit
    else:
        right = right_limit

    left, right = max(left_limit, left), min(right_limit, right)
    if right - left > (2 * max_width + 1):
        left = max(left_limit, peak_idx - max_width)
        right = min(right_limit, peak_idx + max_width + 1)
    return _safe_slice(left, right, len(signal))


def _find_p_wave_interval(signal, qrs_interval, sampling_rate, rhythm_label):
    profile = _get_profile(rhythm_label)
    if not profile["p_present"]:
        return None
    qrs_start = qrs_interval[0]
    if qrs_start < 12:
        return None
    p_wave_ms = profile["p_width_ms"] or (80, 120)
    p_width = _ms_to_samples(int(round(np.mean(p_wave_ms))), sampling_rate)
    search_end = max(qrs_start - _ms_to_samples(25, sampling_rate), 1)
    search_start = max(search_end - _ms_to_samples(180, sampling_rate), 0)
    if search_end - search_start < max(6, p_width // 2):
        return None
    segment = signal[search_start:search_end]
    peak_local = int(np.argmax(np.abs(segment)))
    peak_idx = search_start + peak_local
    if abs(signal[peak_idx]) < 0.02:
        return None
    return _refine_interval_around_peak(
        signal, peak_idx, search_start, search_end, max(6, p_width // 2), ratio=0.30)


def _find_t_wave_interval(signal, qrs_interval, active_end, sampling_rate, rhythm_label):
    profile = _get_profile(rhythm_label)
    if not profile["t_present"]:
        return None
    qrs_end = qrs_interval[1]
    search_start = min(len(signal) - 1, qrs_end + _ms_to_samples(20, sampling_rate))
    search_end = min(active_end, qrs_end + _ms_to_samples(220, sampling_rate))
    if search_end - search_start < 8:
        return None
    segment = signal[search_start:search_end]
    peak_local = int(np.argmax(np.abs(segment)))
    peak_idx = search_start + peak_local
    if abs(signal[peak_idx]) < 0.04:
        return None
    t_width = _ms_to_samples(120, sampling_rate)
    return _refine_interval_around_peak(
        signal, peak_idx, search_start, search_end, max(8, t_width // 2), ratio=0.35)


# =============================================================================
# FEATURE EXTRACTION & DATA LOADING
# =============================================================================
def _process_mask_chunk(
    signals_chunk: np.ndarray,
    labels_chunk: np.ndarray | None,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a contiguous chunk of samples — designed for parallel workers."""
    n, length = signals_chunk.shape
    masks = np.zeros((n, length), dtype=np.uint8)
    fallback_half = int((300 / 1000.0) * sampling_rate / 2)
    fallbacks = np.zeros(n, dtype=bool)
    for i in range(n):
        rhythm_label = int(labels_chunk[i]) if labels_chunk is not None else None
        signal = np.asarray(signals_chunk[i], dtype=float).reshape(-1)
        active_end = _find_active_end(signal)
        peak = _find_dominant_qrs_peak(signal, active_end)
        if peak is None:
            fallbacks[i] = True
            center = length // 2
            _mark_segment(masks[i], center - fallback_half, center + fallback_half)
            continue
        qrs = _build_qrs_interval_from_peak(peak, length, sampling_rate, rhythm_label)
        _mark_segment(masks[i], qrs[0], qrs[1])
        p = _find_p_wave_interval(signal, qrs, sampling_rate, rhythm_label)
        if p is not None:
            _mark_segment(masks[i], p[0], p[1])
        t = _find_t_wave_interval(signal, qrs, active_end, sampling_rate, rhythm_label)
        if t is not None:
            _mark_segment(masks[i], t[0], t[1])
        if masks[i].sum() == 0:
            fallbacks[i] = True
            center = length // 2
            _mark_segment(masks[i], center - fallback_half, center + fallback_half)
    return masks, fallbacks


_CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Paced/Unknown",
}


def fast_clinical_masks(
    signals: np.ndarray, labels: np.ndarray | None, config: PipelineConfig
) -> np.ndarray:
    N, length = signals.shape
    print(f"  Generating clinical masks for {N} samples...")

    # Determine number of workers (leave one core free for OS)
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        lbl_chunk = labels[start:end] if labels is not None else None
        chunks.append((signals[start:end], lbl_chunk))

    masks = np.zeros((N, length), dtype=np.uint8)
    fallbacks = np.zeros(N, dtype=bool)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_mask_chunk, sig_chunk, lbl_chunk, config.sampling_rate
            ): idx
            for idx, (sig_chunk, lbl_chunk) in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            chunk_masks, chunk_fallbacks = future.result()
            start = idx * chunk_size
            masks[start : start + len(chunk_masks)] = chunk_masks
            fallbacks[start : start + len(chunk_fallbacks)] = chunk_fallbacks

    total_fails = int(fallbacks.sum())
    print(f"  [OK] Mask generation complete ({n_workers} workers):")
    print(f"      Total Fallbacks: {total_fails}/{N} ({total_fails / max(N, 1) * 100:.1f}%)")
    
    if labels is not None:
        print("      Per-class fallback rates:")
        for cls in range(config.num_classes):
            cls_mask = labels == cls
            cls_total = int(cls_mask.sum())
            if cls_total > 0:
                cls_fails = int(fallbacks[cls_mask].sum())
                cls_pct = cls_fails / cls_total * 100.0
                print(f"        Class {cls} ({_CLASS_NAMES.get(cls, str(cls))}): {cls_fails}/{cls_total} ({cls_pct:.1f}%)")
            else:
                print(f"        Class {cls} ({_CLASS_NAMES.get(cls, str(cls))}): 0/0 (0.0%)")

    coverage = masks.mean(axis=1)
    print(f"      Coverage - mean: {coverage.mean():.1%}, "
          f"min: {coverage.min():.1%}, max: {coverage.max():.1%}")
    return masks


def extract_tabular_features(
    signals: np.ndarray, labels: np.ndarray | None, config: PipelineConfig
) -> np.ndarray:
    features = np.zeros((len(signals), 3), dtype=np.float32)
    for idx, signal in enumerate(signals):
        sig = np.asarray(signal, dtype=float).reshape(-1)
        label = int(labels[idx]) if labels is not None else None
        active_end = _find_active_end(sig)
        peak = _find_dominant_qrs_peak(sig, active_end)
        if peak is None:
            features[idx, 1] = 0.08
            continue
        qrs = _build_qrs_interval_from_peak(peak, len(sig), config.sampling_rate, label)
        p_wave = _find_p_wave_interval(sig, qrs, config.sampling_rate, label)
        t_wave = _find_t_wave_interval(sig, qrs, active_end, config.sampling_rate, label)
        features[idx, 0] = 1.0 if p_wave is not None else 0.0
        features[idx, 1] = (qrs[1] - qrs[0]) / float(config.sampling_rate)
        features[idx, 2] = 1.0 if t_wave is not None else 0.0
    return features


def validate_signal(raw_signal: np.ndarray, sample_idx: int, label: int, config: PipelineConfig):
    signal = np.asarray(raw_signal, dtype=float).reshape(-1)
    mask = np.zeros(len(signal), dtype=np.uint8)
    active_end = _find_active_end(signal)
    intervals = {"p_wave": [], "qrs": [], "t_wave": []}
    success = False

    peak = _find_dominant_qrs_peak(signal, active_end)
    if peak is not None:
        qrs = _build_qrs_interval_from_peak(peak, len(signal), config.sampling_rate, int(label))
        intervals["qrs"] = [qrs]
        _mark_segment(mask, qrs[0], qrs[1])
        p = _find_p_wave_interval(signal, qrs, config.sampling_rate, int(label))
        if p is not None:
            intervals["p_wave"] = [p]
            _mark_segment(mask, p[0], p[1])
        t = _find_t_wave_interval(signal, qrs, active_end, config.sampling_rate, int(label))
        if t is not None:
            intervals["t_wave"] = [t]
            _mark_segment(mask, t[0], t[1])
        success = True

    validation = {
        "success": success,
        "mask": mask,
        "intervals": intervals,
        "rpeaks": [peak] if peak is not None else [],
        "active_end": int(active_end),
        "profile": _get_profile(int(label))["name"],
        "error": None if success else "No plausible QRS complex detected.",
    }
    return {
        "validation": validation,
        "mask": mask.copy(),
        "qrs_mask": np.zeros_like(mask),
    }


def load_mitbih(config: PipelineConfig):
    if not config.train_path.exists() or not config.test_path.exists():
        raise FileNotFoundError(
            f"MIT-BIH CSVs were not found in {config.data_dir}. "
            "Place mitbih_train.csv and mitbih_test.csv in /data, or pass --data-dir."
        )

    train_df = pd.read_csv(config.train_path, header=None)
    test_df = pd.read_csv(config.test_path, header=None)
    validate_mitbih_frame(train_df, "mitbih_train.csv", config)
    validate_mitbih_frame(test_df, "mitbih_test.csv", config)

    X_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = train_df.iloc[:, -1].to_numpy(dtype=np.int64)
    X_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_df.iloc[:, -1].to_numpy(dtype=np.int64)
    return X_train, y_train, X_test, y_test


# =============================================================================
# DATA PREPROCESSING & AUGMENTATION
# =============================================================================
def central_crop(X: np.ndarray, config: PipelineConfig) -> np.ndarray:
    return X[:, config.crop_start : config.crop_end].astype(np.float32, copy=False)


def per_sample_normalize(X: np.ndarray) -> np.ndarray:
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0.0] = 1.0
    return ((X - mins) / ranges).astype(np.float32)


def apply_tukey_window(X: np.ndarray, config: PipelineConfig) -> np.ndarray:
    window = tukey(config.cropped_len, alpha=config.tukey_alpha).astype(np.float32)
    return (X * window).astype(np.float32)


def apply_clinical_background_mask(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    masks = fast_clinical_masks(X_train, y_train, config).astype(np.float32)
    X_out = X_train.copy()
    n_masked = int(round(config.clinical_mask_fraction * len(X_out)))
    if n_masked > 0:
        indices = rng.choice(len(X_out), size=n_masked, replace=False)
        X_out[indices] = X_out[indices] * masks[indices]
    return X_out.astype(np.float32), masks.astype(np.uint8)


def augment_minority_classes(
    X: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    rng: np.random.Generator,
    M: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    X_parts = [X]
    F_parts = [F]
    y_parts = [y]
    M_parts = [M] if M is not None else None

    for cls, factor in config.augmentation_factors:
        cls_mask = y == cls
        X_cls = X[cls_mask]
        F_cls = F[cls_mask]
        M_cls = M[cls_mask] if M is not None else None
        if len(X_cls) == 0:
            continue

        n_cls = len(X_cls)
        for _ in range(factor - 1):
            X_aug = X_cls.copy()
            M_aug = M_cls.copy() if M_cls is not None else None
            # Vectorized shift: generate all shifts at once, apply via np.roll
            shifts = rng.integers(
                -config.max_shift, config.max_shift + 1, size=n_cls
            )
            for row in range(n_cls):
                if shifts[row] != 0:
                    X_aug[row] = np.roll(X_aug[row], int(shifts[row]))
                    if M_aug is not None:
                        M_aug[row] = np.roll(M_aug[row], int(shifts[row]))
            # Vectorized amplitude jitter
            scales = rng.uniform(0.85, 1.15, size=(n_cls, 1)).astype(
                np.float32
            )
            X_aug = np.clip(X_aug * scales, 0.0, 1.0)

            X_parts.append(X_aug)
            F_parts.append(F_cls.copy())
            y_parts.append(np.full(n_cls, cls, dtype=y.dtype))
            if M_parts is not None:
                M_parts.append(M_aug)

    M_out = np.concatenate(M_parts).astype(np.uint8) if M_parts is not None else None
    return (
        np.concatenate(X_parts).astype(np.float32),
        np.concatenate(F_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        M_out,
    )


def apply_modality_dropout(
    F: np.ndarray, config: PipelineConfig, rng: np.random.Generator
) -> np.ndarray:
    F_out = F.copy().astype(np.float32)
    n_drop = int(round(config.modality_dropout_rate * len(F_out)))
    if n_drop > 0:
        indices = rng.choice(len(F_out), size=n_drop, replace=False)
        F_out[indices] = 0.0
    return F_out


# =============================================================================
# DATASET PREPARATION PIPELINE
# =============================================================================
def prepare_datasets(config: PipelineConfig) -> DatasetBundle:
    rng = np.random.default_rng(config.random_seed)
    X_train_full, y_train_full, X_test_full, y_test = load_mitbih(config)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        central_crop(X_train_full, config),
        y_train_full,
        test_size=config.validation_size,
        stratify=y_train_full,
        random_state=config.random_seed,
    )
    X_test_raw = central_crop(X_test_full, config)

    # -- Normalize FIRST, extract clinical features BEFORE Tukey window --
    # The Tukey taper distorts edge amplitudes and biases P-wave detection
    # thresholds, so features must be measured on the un-tapered signal.
    X_train_norm = per_sample_normalize(X_train_raw)
    X_val_norm = per_sample_normalize(X_val_raw)
    X_test_norm = per_sample_normalize(X_test_raw)

    F_train_clean = extract_tabular_features(X_train_norm, y_train, config)
    F_val = extract_tabular_features(X_val_norm, y_val, config)
    F_test = extract_tabular_features(X_test_norm, y_test, config)

    # -- Generate RRR clinical masks on pre-Tukey normalized signals --
    # Masks are morphological (P/QRS/T regions) and work best on undistorted
    # signals.  They are only needed when rrr_lambda > 0.
    if config.rrr_lambda > 0:
        masks_train_base = fast_clinical_masks(X_train_norm, y_train, config)
        masks_val = fast_clinical_masks(X_val_norm, y_val, config)
    else:
        masks_train_base = None
        masks_val = None

    # -- Apply Tukey window AFTER feature extraction --
    X_train_clean = apply_tukey_window(X_train_norm, config)
    X_val = apply_tukey_window(X_val_norm, config)
    X_test = apply_tukey_window(X_test_norm, config)

    X_train_masked, _ = apply_clinical_background_mask(
        X_train_clean, y_train, config, rng
    )
    X_train_aug, F_train_aug, y_train_aug, masks_train_aug = augment_minority_classes(
        X_train_masked, F_train_clean, y_train, config, rng, M=masks_train_base
    )
    F_train_model = apply_modality_dropout(F_train_aug, config, rng)

    return DatasetBundle(
        X_train=X_train_aug[..., None],
        F_train=F_train_aug,
        y_train=y_train_aug,
        X_train_clean=X_train_clean[..., None],
        F_train_clean=F_train_clean,
        F_train_model=F_train_model,
        X_val=X_val[..., None],
        F_val=F_val,
        y_val=y_val,
        X_test=X_test[..., None],
        F_test=F_test,
        y_test=y_test,
        X_test_raw=X_test_raw.astype(np.float32),
        clinical_masks_train=masks_train_aug,
        clinical_masks_val=masks_val,
    )
