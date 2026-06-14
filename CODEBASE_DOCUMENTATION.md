# ECG XAI Pipeline — Comprehensive Codebase Documentation

> **Generated:** 2026-05-25 
> **Repository root:** `C:\Users\Admin\Desktop\arch`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Directory Layout](#3-directory-layout)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Running the System](#5-running-the-system)
6. [Entry Point: `run_pipeline.py`](#6-entry-point-run_pipelinepy)
7. [Configuration: `ecg_xai_pipeline/config.py`](#7-configuration-ecg_xai_pipelineconfigpy)
8. [Data Layer: `ecg_xai_pipeline/data.py`](#8-data-layer-ecg_xai_pipelinedatapy)
9. [Model Layer: `ecg_xai_pipeline/model.py`](#9-model-layer-ecg_xai_pipelinemodelpy)
10. [Ontology Layer: `ecg_xai_pipeline/ontology.py`](#10-ontology-layer-ecg_xai_pipelineontologypy)
11. [Orchestration: `ecg_xai_pipeline/pipeline.py`](#11-orchestration-ecg_xai_pipelinepipelinepy)
12. [Web Dashboard: `webapp/`](#12-web-dashboard-webapp)
13. [Data Files and Formats](#13-data-files-and-formats)
14. [Output Artifacts](#14-output-artifacts)
15. [Design Decisions](#15-design-decisions)
16. [Appendices](#16-appendices)

---

## 1. Executive Summary

This repository implements a **clinically guided MIT-BIH arrhythmia classification pipeline** with:

- A **hybrid deep learning model** (dilated TCN + CBAM attention + clinical feature branch)
- **Three XAI methods**: SHAP (GradientExplainer), Grad-CAM, Integrated Gradients
- **Inline clinical morphology engine** (P/QRS/T detection and masks — no external service required at runtime)
- **Neuro-symbolic reporting** via a baked-in OWL/RDF ontology and SPARQL queries
- A **Streamlit dashboard** to run the pipeline and explore results

### MIT-BIH class mapping

| Label | Class name | Ontology instance IRI suffix |
|------:|------------|------------------------------|
| 0 | Normal Beat | `Inst_NormalBeat` |
| 1 | Supraventricular Ectopic Beat | `Inst_SVB` |
| 2 | Ventricular Ectopic Beat | `Inst_VEB` |
| 3 | Fusion Beat | `Inst_Fusion` |
| 4 | Paced / Unknown Beat | `Inst_PacedUnknown` |

### Checked-in benchmark (`output/reports/experiment_summary.txt`)

| Metric | Value |
|--------|------:|
| Test accuracy | 0.9876 |
| Macro F1 | 0.9264 |
| XAI samples evaluated | 100 |
| Median Dice | 0.7059 |
| Mean Dice | 0.6236 |
| Median alignment score | 0.6625 |
| Mean alignment score | 0.6037 |
| Mean edge focus | 0.0341 |
| Mean coverage delta | 0.0163 |
| Mean inter-method agreement | 0.5286 |
| XAI Fallback Rate | 0.0% |

The checked-in classification report has 21,892 test beats, with weighted F1 0.99 and macro-average F1 0.93.

---

## 2. System Architecture

### 2.1 Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  webapp/app.py          Streamlit UI (4 pages)                   │
│    subprocess ──────────────────────────────┐                    │
└──────────────────────────────────────────────│───────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  run_pipeline.py        argparse → PipelineConfig → run_pipeline │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  ecg_xai_pipeline/pipeline.py   7-step orchestration             │
└───┬──────────┬──────────┬──────────┬─────────────────────────────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
ontology.py  data.py   model.py   (metrics, plots, CSV writers)
    │          │          │
    └────┬─────┴────┬─────┘
         ▼          ▼
      config.py (shared PipelineConfig)
```

### 2.2 Neuro-symbolic explanation path

```
┌─────────────────────────────┐
│ Inputs                      │
├─────────────────────────────┤
│ CROP: Cropped normalized    │
│       beat                  │
│ RAW : Raw test beat window  │
└───────────────┬─────────────┘
                │
     ┌──────────┴───────────┐
     │                      │
     ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────┐
│ Model path            │  │ Clinical path             │
├──────────────────────┤  ├───────────────────────────┤
│ HYB : Hybrid TCN+CBAM │  │ VAL : validate_signal     │
│ PRED: 5-class         │  │       (on RAW)            │
│       prediction      │  │ MASK: Clinical binary     │
└──────────┬───────────┘  │       mask                │
           │              └──────────┬────────────────┘
           │                         │
           │                         │  (VAL contributes features)
           │                         │
           ▼                         ▼
┌──────────────────────┐     ┌───────────────────────────┐
│ XAI methods          │     │ Feature extraction         │
├──────────────────────┤     ├───────────────────────────┤
│ SHAP                 │────▶│ FEAT: extract_xai_clinical │
│ Grad-CAM             │────▶│       _features            │
│ Integrated Gradients │────▶└──────────┬────────────────┘
└──────────────────────┘                ▲
                                       │
                                       │
                                ┌──────┴───────┐
                                │ VAL (from RAW)│
                                └───────────────┘

┌─────────────────────────────────────────────────────────┐
│ Symbolic reporting                                      │
├─────────────────────────────────────────────────────────┤
│ SPARQL: SPARQL beat profile                             │
│ TXT   : rapport_sample_ID.txt                           │
└─────────────────────────────────────────────────────────┘
           ▲                          ▲
           │                          │
           │                          │
           │                    ┌─────┴─────┐
           │                    │ FEAT       │
           │                    └─────┬─────┘
           │                          │
     ┌─────┴─────┐                    │
     │ PRED       │────────────────────┘
     └────────────┘

Overall dependency:
- CROP → HYB → PRED → SPARQL
- HYB → {SHAP, Grad-CAM, Integrated Gradients} → FEAT
- RAW  → VAL → MASK
- VAL → FEAT
- FEAT → SPARQL → TXT
```

### 2.3 Why dual output heads & RRR training

| Head | Source | Loss weight | Role |
|------|--------|------------:|------|
| `main_output` | Fused signal + clinical features | 1.0 | Primary classifier for deployment metrics |
| `aux_output` | Signal branch only (GAP → Dense) | 0.30 | Auxiliary classifier for signal representation training |

*XAI Target Head*: To avoid explanation mismatch, the pipeline now consistently configures and routes SHAP, Grad-CAM, and Integrated Gradients to explain the **main prediction head** (`model.outputs[0]`) taking both inputs, so that the explanations represent what the deployed model actually predicts.

---

## 3. Directory Layout

| Path | Type | Role |
|------|------|------|
| `run_pipeline.py` | Python | CLI entry; builds `PipelineConfig` from flags |
| `requirements.md` | Markdown | Pinned dependency table |
| `ecg_xai_pipeline/config.py` | Python | Immutable experiment configuration |
| `ecg_xai_pipeline/data.py` | Python | Data loading, clinical masks, augmentation |
| `ecg_xai_pipeline/model.py` | Python | Keras model, training, XAI |
| `ecg_xai_pipeline/ontology.py` | Python | RDF ontology + neuro-symbolic reports |
| `ecg_xai_pipeline/pipeline.py` | Python | Full experiment runner |
| `webapp/app.py` | Python | Streamlit dashboard |
| `webapp/overview_hero.html` | HTML | Overview page architecture cards |
| `webapp/.streamlit/config.toml` | TOML | UI theme, port 8501 |
| `data/mitbih_train.csv` | Data | Training beats |
| `data/mitbih_test.csv` | Data | Test beats |
| `output/` | Generated | Model, reports, figures, XAI outputs |

**Package note:** `ecg_xai_pipeline/` is an explicit Python package and re-exports `PipelineConfig` for lightweight imports.

**Private GitHub packaging:** this repository is intended to include source code, the MIT-BIH CSV files under `data/`, the saved Keras model under `output/`, and generated reports/images under `output/`. Dataset CSVs and model binaries are tracked with Git LFS via `.gitattributes`. Local virtual environments, Python caches, editor folders, local environment files, and Streamlit secrets remain ignored.

---

## 4. End-to-End Data Flow

### 4.1 Training path (conceptual)

1. **Load** `mitbih_train.csv` / `mitbih_test.csv` (last column = label).
2. **Crop** each beat to indices 20–167 (147 samples at 125 Hz ≈ 1.18 s).
3. **Normalize** per sample to [0, 1].
4. **Extract features** on normalized signal *before* Tukey window (critical for P-wave thresholds).
5. **Apply Tukey** taper (alpha=0.10) on train/val/test.
6. **Mask 30%** of training signals with clinical binary masks (background suppression).
7. **Augment** minority classes (shift ±15 samples, scale 0.85–1.15).
8. **Modality dropout** on 20% of clinical feature vectors (zeros all 3 features).
9. **Add channel** dimension `(N, 147, 1)`.
10. **Train** hybrid model with class-weighted sampling and early stopping.

### 4.2 Inference + XAI path

1. Load best checkpoint from `output/ecg_tcn_cbam_hybrid.keras`.
2. Predict on full test set → confusion matrix + classification report.
3. Select `samples_per_class` (default 20) indices per class from test labels.
4. SHAP background from random training subset (`shap_background_size` default 200).
5. For each sample: attributions → smoothed importance → thresholded mask.
6. Compare XAI mask vs `validate_signal` mask on the normalized central test crop before Tukey taper (`X_test_raw`).
7. Write ontology report per sample.

---

## 5. Running the System

### 5.1 Setup

```bash
cd C:\Users\Admin\Desktop\arch
python -m venv .venv
.venv\Scripts\activate
pip install numpy==2.3.5 pandas==2.3.1 scipy==1.16.1 scikit-learn==1.7.1 matplotlib==3.10.3 seaborn==0.13.2 tensorflow==2.20.0 shap==0.50.0 rdflib==7.6.0 streamlit==1.57.0 Pillow==12.2.0 plotly==6.7.0
```

The pinned dependency list is also documented in `requirements.md`.

Place `mitbih_train.csv` and `mitbih_test.csv` under `data/` (or set `MITBIH_DATA_DIR`).

### 5.2 CLI examples

```bash
# Full train + evaluate + XAI (default 20 epochs)
python run_pipeline.py

# Resume with existing weights, skip XAI
python run_pipeline.py --mode resume_post_xai --no-xai

# Heavier XAI sampling
python run_pipeline.py --epochs 30 --samples-per-class 20 --shap-background-size 400

# Custom paths
python run_pipeline.py --data-dir D:\datasets\mitbih --output-dir D:\experiments\run1
```

### 5.3 Dashboard

```bash
streamlit run webapp/app.py
```

Configure run on **Run Pipeline** page; inspect results on **XAI Explorer** and **Reports**.

### 5.4 Environment variables

| Variable | Overrides |
|----------|-----------|
| `MITBIH_DATA_DIR` | `PipelineConfig.data_dir` |
| `MEDICAL_ENGINE_DIR` | `PipelineConfig.medical_engine_dir` (legacy) |
| `TF_DETERMINISTIC_OPS` | Set to `1` by `set_global_seed` |

---

## 6. Entry Point: `run_pipeline.py`

`run_pipeline.py`: CLI → merged `PipelineConfig` → `run_pipeline(config)` (lazy import inside `main`).

### 6.1 `parse_args() -> argparse.Namespace`

| Argument | Type | Default | Effect |
|----------|------|---------|--------|
| `--data-dir` | `Path` | env `MITBIH_DATA_DIR` or `./data` | Training/test CSV location |
| `--medical-engine-dir` | `Path` | env `MEDICAL_ENGINE_DIR` or Desktop path | Reserved external root |
| `--output-dir` | `Path` | `PROJECT_ROOT/output` | All artifacts |
| `--mode` | `full` \| `resume_post_xai` | `full` | Skip training if model exists |
| `--epochs` | `int` | 20 | Training epochs |
| `--batch-size` | `int` | 64 | Fit/predict batch size |
| `--samples-per-class` | `int` | 5 | XAI stratified sample count per class |
| `--shap-background-size` | `int` | 200 | SHAP background set size |
| `--no-xai` | flag | off | Sets `run_xai=False` |

### 6.2 `main() -> None`

`parse_args()` → base `PipelineConfig()` → override only args that are not `None` → `run_xai = not args.no_xai` → `run_pipeline(config)`.

### 6.3 Execution

```bash
python run_pipeline.py
python run_pipeline.py --mode resume_post_xai --no-xai
python run_pipeline.py --epochs 30 --samples-per-class 20
```

---

## 7. Configuration: `ecg_xai_pipeline/config.py`

Frozen `PipelineConfig`: paths, training/XAI hyperparameters, and derived output paths (see tables below).

### 7.1 Module-level constants

| Name | Value | Meaning |
|------|-------|---------|
| `PROJECT_ROOT` | parent of `ecg_xai_pipeline/` | Repository root |
| `DEFAULT_DATA_DIR` | `PROJECT_ROOT / "data"` | Default CSV folder |
| `DEFAULT_EXTERNAL_ROOT` | `~/Desktop/simplified model` | Legacy external medical engine path |

Optional env: `MITBIH_DATA_DIR` → `data_dir`; `MEDICAL_ENGINE_DIR` → `medical_engine_dir`.

### 7.2 `PipelineConfig` dataclass (frozen)

#### Paths and files

| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | `DEFAULT_DATA_DIR` or env | Root for CSVs |
| `medical_engine_dir` | `DEFAULT_EXTERNAL_ROOT` or env | External reference path |
| `output_dir` | `PROJECT_ROOT/output` | Artifact root |
| `train_csv` | `mitbih_train.csv` | Training filename |
| `test_csv` | `mitbih_test.csv` | Test filename |
| `model_name` | `ecg_tcn_cbam_hybrid.keras` | Saved model filename |

#### Signal geometry

| Field | Default | Description |
|-------|---------|-------------|
| `sampling_rate` | 125 | Hz (MIT-BIH beat windows) |
| `num_classes` | 5 | Output softmax size |
| `num_features` | 3 | Clinical tabular features |
| `crop_start` | 20 | Inclusive crop index |
| `crop_end` | 167 | Exclusive crop index |
| `cropped_len` (property) | 147 | `crop_end - crop_start` |

#### Training

| Field | Default |
|-------|---------|
| `validation_size` | 0.15 |
| `random_seed` | 42 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `learning_rate` | 1e-3 |

#### Augmentation / regularization

| Field | Default | Description |
|-------|---------|-------------|
| `clinical_mask_fraction` | 0.30 | Fraction of train beats masked |
| `modality_dropout_rate` | 0.20 | Fraction of samples with zeroed features |
| `tukey_alpha` | 0.10 | Tukey window taper parameter |
| `max_shift` | 15 | Max circular shift for augmentation |
| `augmentation_factors` | `(1,5), (3,5), (2,2), (4,2)` | Per-class copy multipliers |

#### XAI & RRR Regularization

| Field | Default | Description |
|-------|---------|-------------|
| `shap_background_size` | 200 | Background samples for GradientExplainer |
| `shap_use_aux_head` | True | Compatibility flag; explanations are routed to main `outputs[0]` |
| `integrated_gradients_steps` | 64 | IG interpolation steps |
| `integrated_gradients_use_aux_head` | True | Compatibility flag; explanations are routed to main `outputs[0]` |
| `integrated_gradients_smooth_samples` | 4 | Noise smoothing runs |
| `integrated_gradients_noise_std` | 0.01 | Gaussian noise on input |
| `gradcam_layer_name` | `stem2_relu` | Target conv layer (falls back to `gradcam_target`) |
| `gradcam_use_aux_head` | True | Compatibility flag; explanations are routed to main `outputs[0]` |
| `xai_score_mode` | `log_probability` | Gradient scalar: log p(class) |
| `samples_per_class` | 20 | Stratified XAI picks |
| `xai_percentile` | 75.0 | Default mask threshold percentile |
| `xai_match_clinical_coverage` | True | Binary search threshold to match mask area |
| `xai_smoothing_sigma` | 3.5 | Gaussian smooth on |attribution| |
| `xai_zero_edges_before_threshold` | True | Zero first/last 15 importance samples |
| `dilation_radius` | 2 | Morphological dilation on binary mask |
| `ignore_edge` | 15 | Edge samples excluded in similarity metrics |
| `rrr_lambda` | 0.05 | Weight coefficient for Right for the Right Reasons (RRR) regularization |
| `rrr_warmup_epochs` | 3 | Warmup epochs to scale `active_lambda` from 0.0 to `rrr_lambda` |
| `rrr_use_aux_head` | True | Outputs selection for computing gradient penalty during RRR training |

#### Run control

| Field | Default |
|-------|---------|
| `mode` | `"full"` |
| `run_xai` | `True` |

### 7.3 Derived path properties

| Property | Returns |
|----------|---------|
| `train_path` | `data_dir / train_csv` |
| `test_path` | `data_dir / test_csv` |
| `model_path` | `output_dir / model_name` |
| `reports_dir` | `output_dir / reports` |
| `figures_dir` | `output_dir / figures` |
| `xai_dir` | `output_dir / xai` |
| `medical_dir` | `output_dir / medical` |

### 7.4 `ensure_dirs() -> None`

Ensures `data_dir`, `output_dir`, and all artifact subdirs under `output_dir` exist.

---

## 8. Data Layer: `ecg_xai_pipeline/data.py`

CSV load/crop/normalize, inline P/QRS/T morphology, parallel clinical masks, three-value tabular features, augmentation, and `DatasetBundle` for `model` / `pipeline`. Re-exports a few `ontology` symbols for legacy imports.

---

### 8.1 Dataclass: `DatasetBundle`

```python
@dataclass
class DatasetBundle:
    X_train: np.ndarray          # (N, 147, 1) float32 — augmented, possibly masked
    F_train: np.ndarray          # (N, 3) — clinical features (post-augmentation rows)
    y_train: np.ndarray          # (N,) int64 — labels (augmented)
    X_train_clean: np.ndarray    # (N_orig_aug, 147, 1) — Tukey-windowed, unmasked train (pre-aug size differs)
    F_train_clean: np.ndarray    # clean features before modality dropout
    F_train_model: np.ndarray    # (N, 3) — with modality dropout applied
    X_val: np.ndarray            # (N_val, 147, 1)
    F_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray           # (N_test, 147, 1)
    F_test: np.ndarray
    y_test: np.ndarray
    X_test_raw: np.ndarray       # (N_test, 187) — full-width normalized crop BEFORE Tukey (for validate_signal)
```

| Field | Shape (typical) | Purpose |
|-------|-----------------|--------|
| `X_*` | `(N, 147, 1)` | Conv input; `F_*` is `(N, 3)` tabular; `X_test_raw` is normalized pre-Tukey crop for `validate_signal` overlays |

Built only in `prepare_datasets()`; consumed by `model` training/predict and `pipeline` XAI.

---

### 8.2 Data validation

#### `validate_mitbih_frame(df, split_name, config) -> None`

Checks width ≥ `crop_end + 1`, last column labels in `0..num_classes-1`, signal columns finite `float32`. Raises `ValueError` with `split_name` context. Used right after `read_csv` in `load_mitbih`.

---

### 8.3 Class morphology profiles (module-private)

`_CLASS_PROFILES`: per-class `name`, `p_present`, `t_present`, `qrs_ms` range, optional `p_width_ms`. `_get_profile(label)` defaults class `None` → 0. Helpers: `_ms_to_samples`, `_safe_slice`, `_mark_segment` (in-place mask fill).

---

### 8.4 Signal morphology helpers

- **`_find_active_end`**: trims flat padded tail using gradient + low-amplitude heuristics.
- **`_find_dominant_qrs_peak`**: local maxima on `|signal|` inside margin window; scored by amplitude, sharpness, center bias; floor `|amp| >= 0.15`.
- **`_build_qrs_interval_from_peak`**: QRS half-width from profile `qrs_ms`, clamped via `_safe_slice`.
- **`_refine_interval_around_peak`**: expand from peak to baseline-crossing threshold, width-capped (used for P/T).
- **`_find_p_wave_interval` / `_find_t_wave_interval`**: profile-gated search windows before/after QRS; argmax + refine thresholds as in code.

---

### 8.5 Parallel mask generation

**`_process_mask_chunk`:** per row — `active_end` → QRS → optional P/T from profile; on failure, ±150 ms center fallback. Returns masks + fallback count.

**`fast_clinical_masks`:** `ProcessPoolExecutor` over `cpu_count()-1` chunks, merge in order; used by `apply_clinical_background_mask`.

---

### 8.6 Feature extraction and loading

**`extract_tabular_features`:** `(N, 3)` — P present (0/1), QRS duration (s, default 0.08 if missing), T present (0/1). Run before Tukey in `prepare_datasets`.

**`validate_signal`:** `raw_signal` (typically `X_test_raw` row); `label` = **predicted** class for the morphology profile. Returns `validation` (success, mask, intervals, `active_end`, …) plus a top-level `mask` copy; `qrs_mask` is reserved.

**`load_mitbih`:** full-width train/test `X`, `y` for `prepare_datasets` only.

---

### 8.7 Preprocessing and augmentation

| Function | Role |
|----------|------|
| `central_crop` | `crop_start:crop_end` → 147 samples |
| `per_sample_normalize` | Row min–max to [0, 1] |
| `apply_tukey_window` | Row-wise Tukey(`tukey_alpha`) |
| `apply_clinical_background_mask` | `fast_clinical_masks` then multiply random `clinical_mask_fraction` of train rows by mask |
| `augment_minority_classes` | Per `augmentation_factors`: copies with shift ±`max_shift`, scale 0.85–1.15 |
| `apply_modality_dropout` | Zeros full feature row at `modality_dropout_rate` |

---

### 8.8 `prepare_datasets(config) -> DatasetBundle`

Order matters: `load_mitbih` → stratified val split → crop → normalize → **`extract_tabular_features` (before Tukey)** → Tukey → train-only clinical mask → minority augment → modality dropout → `F_train_model` → add channel dim `[..., None]`. Invoked from `pipeline.run_pipeline`.

---

## 9. Model Layer: `ecg_xai_pipeline/model.py`

Hybrid **TCN + CBAM** Keras model (main + aux softmax), class-weighted training, SHAP / Grad-CAM / Integrated Gradients. Used only from `pipeline` (not `webapp`).

### 9.1 Architecture blocks

**`conv_block`:** Conv1D → BN → ReLU → optional MaxPool (stems).

**`tcn_residual_block`:** dilated Conv1D ×2 + residual (1×1 adjust if needed); stack dilations `1,2,4,8,16` @ 128 filters, dropout 0.10.

**`cbam_attention`:** channel gate (avg/max pool → Dense) + temporal `Conv1D(1, sigmoid)`; names `cbam_multiply`, etc., used for Grad-CAM.

**`build_hybrid_tcn_cbam_model`:** inputs `ecg_signal` `(147,1)` and `clinical_features` `(3,)`.

```
Input → stem1 → stem2 → Dropout(0.1)
     → TCN ×5 (d=1,2,4,8,16)
     → Conv1D(128, relu) [gradcam_target]
     → CBAM → LayerNorm → GAP → Dense(64) → Dropout(0.2)
     → aux_output (softmax, num_classes)
```

**Feature branch:** `Dense(16) → Dense(16)`.

**Fusion:** `Concatenate(signal_repr, feature_repr) → Dense(64) → Dropout(0.3) → main_output (softmax)`.

**Outputs:** `[main_output, aux_output]` — saved as `ecg_tcn_cbam_hybrid`.

### 9.2 Training utilities

**`set_global_seed`:** NumPy + Keras seeds and `TF_DETERMINISTIC_OPS=1`.

**`class_sample_weights`:** `compute_class_weight("balanced")`, clipped to `[1, 10]`, expanded to per-sample weights.

**`compile_model`:** Adam(`learning_rate`); sparse categorical CE on both heads (weights 1.0 / 0.30); accuracy per head.

**`RRRModel`:** Custom subclassed wrapper that intercepts Keras `train_step` and `test_step` to add the input gradient penalty term (Right for the Right Reasons) outside the clinical masks. Features a monkey-patched `output_names` property to support dict-based compilation in Keras.

**`EpochTrackerCallback` & `FunctionalModelCheckpoint`:** Callbacks used with `RRRModel` to track epochs (for linear lambda warmup) and checkpoint the raw functional base model directly.

**`train_or_load_model`:** If `mode=resume_post_xai` and `model_path` exists → `load_model(compile=False)`; else build model and, if `config.rrr_lambda > 0`, wrap it in `RRRModel` and train with training/validation clinical masks and custom callbacks; otherwise compile standard model and fit on `[X_train, F_train_model]` with ModelCheckpoint to `config.model_path`.

### 9.3 Prediction

**`predict_main`:** `(main_probs (N,5), y_pred)` from main head.

**`select_stratified_samples`:** up to `samples_per_class` random indices per class, sorted.

### 9.4 XAI helper models

**`main_output_model`:** single-output wrapper.

**`explanation_output_model(use_aux_head)`:** Returns a model explaining the main output (`model.outputs[0]`) using all inputs to fix the XAI target-head explanation mismatch.

### 9.5 Attribution methods

**SHAP:** `GradientExplainer` on `explanation_output_model` (main predictions head, training background) → `_extract_signal_shap`.

**Grad-CAM:** submodel `[conv, preds]`; class-score gradient w.r.t. activations of the main predictions head (`model.outputs[0]`) → weighted CAM → `np.interp` to signal length; optional input-gradient gate.

**Integrated Gradients:** zero baseline, α interpolation, averaged noisy runs per config → channel-0 attribution.

**`compute_all_attributions`:** convenience dict; `pipeline` still calls the three functions directly with the same flags.

---

### 9.6 Model summary table

| Component | Hyperparameter |
|-----------|----------------|
| Classes | 5 (MIT-BIH) |
| Input length | 147 samples @ 125 Hz |
| Clinical features | 3 |
| TCN filters | 128 |
| CBAM reduction | 8 |
| Aux loss weight | 0.30 |
| XAI target head | aux (default) |
| Grad-CAM layer | `stem2_relu` (config; code fallback `gradcam_target`) |

---

## 10. Ontology Layer: `ecg_xai_pipeline/ontology.py`

Baked TTL (`BAKED_IN_ONTOLOGY_TTL`), optional rdflib graph, SPARQL, XAI feature packaging, and text reports consumed by `pipeline`. Some symbols are re-exported from `data.py` for older import paths.

### 10.1 Ontology content (`BAKED_IN_ONTOLOGY_TTL`)

**Namespace:** `http://www.research.org/ontology/ecg#` (`ecg:`)

**Class hierarchy:**

| Class | Parent |
|-------|--------|
| `ECG_Signal` | (root beat types) |
| `ECG_Component` | — |
| `Wave` | `ECG_Component` |
| `Complex` | `ECG_Component` |
| `P_Wave`, `T_Wave` | `Wave` |
| `QRS_Complex` | `Complex` |
| Beat types | `ECG_Signal` subclasses |

**MIT-BIH beat instances (linked to components):**

| Class ID | Instance IRI | OWL class |
|--------:|--------------|-----------|
| 0 | `Inst_NormalBeat` | `NormalBeat` |
| 1 | `Inst_SVB` | `SupraventricularEctopicBeat` |
| 2 | `Inst_VEB` | `VentricularEctopicBeat` |
| 3 | `Inst_Fusion` | `FusionBeat` |
| 4 | `Inst_PacedUnknown` | `PacedUnknownBeat` |

**Datatype properties:** `minDurationMs`, `maxDurationMs`, `morphology`, `presence`, `minAmplitudeMv`, `maxAmplitudeMv`, `prIntervalMinMs`, `prIntervalMaxMs`, `qtIntervalMaxMs`.

**Object property:** `hasComponent` (domain `ECG_Signal`, range `ECG_Component`).

---

### 10.2 Loading and export

**`OntologyStatus`:** `enabled`, graph stats, `message`.

**`load_baked_ontology`:** LRU-cached parse of TTL; on missing rdflib → `(None, disabled status)`.

**`ontology_summary_text`:** written to `output/reports/ontology_status.txt`.

**`component_duration_rows`:** SPARQL over P/T/QRS components → rows for `ontology_component_durations.csv` when a graph exists.

### 10.3 Class IRI maps (module-private)

`_CLASS_TO_BEAT_IRI` and `_CLASS_TO_BEAT_NAME`: map MIT-BIH `0..4` to SPARQL instance IRIs and display strings (aligned with the executive summary table).

---

### 10.4 XAI clinical feature extraction

**`extract_xai_clinical_features`:** from `validate_signal` bundle + per-method importance + raw row → `qrs_duration_ms`, `focus_region` (consensus XAI vs P/QRS/T masks), `t_wave_morphology` (sign / flat / absent). **`_classify_t_wave`**, **`_consensus_focus_region`:** helpers (normalize curves, mean consensus, pick dominant region).

### 10.5 Neuro-symbolic inference

**`infer_ontology_report`:** `_sparql_beat_profile` → `_build_clinical_narrative` (focus, QRS vs ontology, P/T clauses, conclusion) → fixed-width `NEURO-SYMBOLIC CLINICAL REPORT` → `output/xai/rapport_sample_{idx}.txt`.

**`_sparql_beat_profile`:** parameterized query on beat IRI; empty graph → class-name fallback. **`_qrs_assessment`:** within / narrower / wider vs ontology QRS bounds. Narrative slots are driven by TTL so ontology edits propagate without Python edits.

---

## 11. Orchestration: `ecg_xai_pipeline/pipeline.py`

End-to-end runner: ontology status → `prepare_datasets` → train/load → predict/metrics → optional XAI + clinical comparison + plots + CSV/TXT. Entry: `run_pipeline(config)` from the CLI script.

### 11.1 Constants

#### `CLASS_NAMES`

```python
{0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion", 4: "Paced/Unknown"}
```

Used in classification reports and plot titles.

---

### 11.2 Metrics and mask utilities

| Symbol | Role |
|--------|------|
| `classification_summary` | accuracy / F1 / sklearn report string |
| `attribution_to_importance` | abs + optional Gaussian smooth + normalize |
| `suppress_edge_importance` | zero `ignore_edge` samples at ends |
| `importance_to_binary_mask` | percentile threshold or binary-search to match clinical coverage |
| `mask_overlap_score`, `focused_similarity`, `inter_method_agreement` | standardized XAI-vs-clinical alignment and inter-method agreement metrics |
| `save_dataframe` | CSV writer helper |
| `_comparison_is_valid` | requires successful validation, mask, QRS intervals |

### 11.3 Visualization

Matplotlib Agg + seaborn; figures under `figures/` and per-sample XAI PNGs under `xai/` (confusion matrix, overlays, method comparison, metric summary, inter-method agreement).

### 11.4 `run_pipeline(config) -> dict`

| Step | Action | Key outputs |
|------|--------|-------------|
| [1/7] | `load_baked_ontology` | `ontology_status.txt`, `ontology_component_durations.csv` |
| [2/7] | `prepare_datasets` | `DatasetBundle` |
| [3/7] | `train_or_load_model` | `ecg_tcn_cbam_hybrid.keras` |
| [4/7] | `predict_main` + metrics | `classification_report.txt`, `confusion_matrix.png` |
| [5/7] | XAI (if `run_xai`) | SHAP, GradCAM, IG |
| [6/7] | Per-sample evaluation | `xai_clinical_comparison.csv`, PNGs, `rapport_sample_*.txt` |
| [7/7] | Summary | `experiment_summary.txt` |

If `run_xai=False`, stops after step 4 (`summary`, `elapsed`).

**Per XAI sample:** `validate_signal(X_test_raw, predicted class)` → each method: importance → mask (coverage-matched when clinical mask valid) → alignment/coverage diagnostics, plots, CSV → `extract_xai_clinical_features` + `infer_ontology_report`; aggregate inter-method agreement rows.

**Returns:** `{ "summary", "xai" (DataFrame), "agreement" (DataFrame), "elapsed" }` on full run.

---

## 12. Web Dashboard: `webapp/`

Streamlit app (`webapp/app.py`): overview, subprocess launcher for `run_pipeline.py`, XAI explorer, reports. No `ecg_xai_pipeline` imports — reads/writes `output/` like the CLI. Assets: `overview_hero.html`, `.streamlit/config.toml` (theme, port 8501).

### 12.1 Path constants

| Name | Resolves to |
|------|-------------|
| `PROJECT_ROOT` | Parent of `webapp/` (repo root) |
| `OUTPUT_DIR` | `PROJECT_ROOT/output` |
| `REPORTS_DIR` | `output/reports` |
| `FIGURES_DIR` | `output/figures` |
| `XAI_DIR` | `output/xai` |
| `MEDICAL_DIR` | `output/medical` |
| `RUNNER_SCRIPT` | `run_pipeline.py` |

#### `CLASS_NAMES` (UI labels)

Slightly different strings than pipeline (`"Paced / Unknown"` vs `"Paced/Unknown"`).


---

### 12.3 Pages

- **`page_overview`:** embeds `overview_hero.html`.
- **`page_run_pipeline`:** widgets map to CLI flags (`--mode`, `--epochs`, `--batch-size`, `--samples-per-class`, `--shap-background-size`, `--no-xai`); runs `subprocess.Popen` on `RUNNER_SCRIPT`, streams stdout into session state, optional stop + periodic `st.rerun` while alive.
- **`page_xai_explorer`:** sample picker from SHAP PNGs; cards/tabs read `xai_clinical_comparison.csv`, per-method PNGs, method comparison + summary figures, ontology `rapport_sample_*.txt`, optional `medical/` images.
- **`page_reports`:** text reports and CSVs with downloads.

### 12.4 `main()`

`st.set_page_config` + `st.navigation` (Overview, Run Pipeline, XAI Explorer, Reports). Sidebar pill shows whether `OUTPUT_DIR` has content.

```bash
streamlit run webapp/app.py
```

---

## 13. Data Files and Formats

### 13.1 MIT-BIH CSV layout

| Property | Value |
|----------|-------|
| Files | `data/mitbih_train.csv`, `data/mitbih_test.csv` |
| Header | None (`header=None` in pandas) |
| Columns | 187 signal samples + 1 label column |
| Label range | 0–4 (5-class problem) |
| Sampling | 125 Hz beat-centered windows |

### 13.2 Cropping geometry

| Index | Value |
|-------|-------|
| `crop_start` | 20 (inclusive) |
| `crop_end` | 167 (exclusive) |
| `cropped_len` | 147 samples ≈ 1.176 s |

`X_test_raw` in `DatasetBundle` stores **normalized** central crop **before** Tukey (same index range, no extra channel).

### 13.3 Clinical feature vector

| Dim | Name | Range / type |
|-----|------|----------------|
| 0 | P-wave detected | 0.0 or 1.0 |
| 1 | QRS duration (seconds) | ~0.08–0.2 |
| 2 | T-wave detected | 0.0 or 1.0 |

---

## 14. Output Artifacts

### 14.1 Directory tree (after full run)

```
output/
├── ecg_tcn_cbam_hybrid.keras
├── reports/
│   ├── ontology_status.txt
│   ├── ontology_component_durations.csv
│   ├── classification_report.txt
│   ├── experiment_summary.txt
│   ├── xai_clinical_comparison.csv
│   └── xai_inter_method_agreement.csv
├── figures/
│   ├── confusion_matrix.png
│   ├── xai_method_metric_summary.png
│   └── xai_inter_method_agreement.png
└── xai/
    ├── shap_sample_{id}.png
    ├── gradcam_sample_{id}.png
    ├── integratedgradients_sample_{id}.png
    ├── method_comparison_sample_{id}.png
    └── rapport_sample_{id}.txt
```

### 14.2 Key CSV schemas

#### `xai_clinical_comparison.csv`

| Column | Description |
|--------|-------------|
| `sample_idx` | Test set index |
| `method` | SHAP / GradCAM / IntegratedGradients |
| `y_true`, `y_pred` | Ground truth vs prediction |
| `pred_confidence` | Softmax at predicted class |
| `comparison_valid` | Clinical mask usable |
| `dice`, `alignment_score`, `alignment_grade`, `edge_focus` | Raw overlap, edge-aware alignment, and diagnostic |
| `threshold`, `threshold_policy` | Binarization details |
| `xai_coverage`, `clinical_coverage`, `coverage_delta` | Mask areas and coverage mismatch |

#### `xai_inter_method_agreement.csv`

| Column | Description |
|--------|-------------|
| `sample_idx` | Test index |
| `method_a`, `method_b` | Pair |
| `agreement_score` | Inter-method overlap |

### 14.3 `experiment_summary.txt` keys

`accuracy`, `macro_f1`, `xai_samples`, `xai_fallback_count`, `xai_fallback_rate`, `median_dice`, `mean_dice`, `median_alignment_score`, `mean_alignment_score`, `mean_edge_focus`, `mean_coverage_delta`, `mean_inter_method_agreement`, `elapsed`.

---

## 16. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **TCN + CBAM** instead of Transformer | Local convolutions + dilations give receptive field without O(n²) attention; Grad-CAM-friendly |
| **Dual heads (main + aux)** | Aux head uses signal branch only for aux representation loss training; main fuses clinical features for final classification accuracy |
| **Explain Main Head** | Explains the main fused output (`model.outputs[0]`) containing clinical features because it is the target deployed model actually predicts, fixing target mismatch |
| **RRR Regularization** | Penalizes input gradients w.r.t signal outside the clinical morphology wave masks (`rrr_lambda = 0.05`) to enforce clinical fidelity during TCN training |
| **RRR Training Wrapper** | Implements custom train loop inside Keras `RRRModel` subclass but checkpoints base Functional model to keep prediction pipelines clean and dependency-free |
| **Aligned Edge Suppression** | Clinical mask edges are suppressed symmetrically to match the XAI edge-suppressed domain before computing the standardized alignment score, preventing boundary artifacts |
| **Inline morphology** | No runtime dependency on external medical engine; reproducible masks in `data.py` |
| **Features before Tukey** | Edge taper corrupts P-wave amplitude thresholds |
| **Class-weighted training** | MIT-BIH imbalance; weights capped at 10× |
| **Coverage-matched thresholds** | Fair alignment comparison when clinical mask area differs from XAI percentile default |
| **Baked ontology** | No external OWL file required; SPARQL drives narrative text |
| **Subprocess in Streamlit** | Long TensorFlow jobs avoid blocking Streamlit interpreter; logs streamed live |
| **Frozen `PipelineConfig`** | Prevents accidental mutation mid-run |

---

## 17. Appendices

### Appendix F — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` for CSVs | Missing `data/*.csv` | Copy MIT-BIH exports or set `--data-dir` |
| Ontology `enabled=False` | rdflib not installed | `pip install rdflib==7.6.0` |
| SHAP slow / OOM | Large background set | Reduce `--shap-background-size` |
| All XAI alignment scores near 0 | Invalid clinical detection | Check `comparison_valid` column; inspect raw signal scale |
| Resume still retrains | Model path missing | Ensure `output/ecg_tcn_cbam_hybrid.keras` exists |
| Streamlit shows no outputs | Pipeline never completed | Run CLI or dashboard launcher first |
| Import errors for package | Wrong working directory | Run from repo root: `python run_pipeline.py` |

### Appendix G — Glossary

| Term | Meaning |
|------|---------|
| **MIT-BIH** | Arrhythmia database; 5-class beat labels used here |
| **TCN** | Temporal Convolutional Network — dilated causal-style conv stacks |
| **CBAM** | Convolutional Block Attention Module — channel + temporal gates |
| **XAI** | Explainable AI — SHAP, Grad-CAM, Integrated Gradients |
| **Clinical mask** | Binary mask from P/QRS/T heuristics (`validate_signal` / `fast_clinical_masks`) |
| **Aux head** | Signal-only softmax used for attribution by default |
| **Neuro-symbolic** | Neural predictions + ontology SPARQL narrative |
| **Modality dropout** | Regularization zeroing clinical feature vectors |
| **Tukey window** | Tapered cosine edge window reducing spectral leakage |

---

*End of codebase documentation. Last revised 2026-05-26.*
