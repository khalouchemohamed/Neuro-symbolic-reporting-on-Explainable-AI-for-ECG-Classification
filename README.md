# Neuro-Symbolic Reporting on Explainable AI for ECG Classification

Private research project for MIT-BIH ECG beat classification with a hybrid TCN/CBAM neural model, clinical feature fusion, Right-for-the-Right-Reasons regularization, explainable AI overlays, and ontology-backed neuro-symbolic reports.

The repository intentionally includes the local experiment artifacts:

- `data/mitbih_train.csv`
- `data/mitbih_test.csv`
- `output/ecg_tcn_cbam_hybrid.keras`
- `output/figures/*.png`
- `output/reports/*`
- `output/xai/*.png` and `output/xai/rapport_sample_*.txt`

Large data/model files are tracked with Git LFS.

## Current Results

The checked-in `output/reports/experiment_summary.txt` reports:

| Metric | Value |
|---|---:|
| Test Accuracy (Baseline) | 0.9876 |
| Macro F1-Score (Baseline) | 0.9264 |
| Occluded Test Accuracy | 52.54% |
| Absolute Accuracy Drop | 46.23% |
| Balanced Accuracy Drop | 51.82% |
| Median Clinical Dice | 0.7059 |
| Median Alignment Score | 0.6625 |
| Mean Inter-Method Dice | 0.5286 |
| Mean Edge Focus | 0.0341 |

## Project Layout

```text
run_pipeline.py                       CLI entry point
requirements.md                       Pinned dependency table
CODEBASE_DOCUMENTATION.md             Detailed architecture and implementation notes

ecg_xai_pipeline/
  __init__.py                         Package export for PipelineConfig
  config.py                           Reproducible experiment configuration
  data.py                             CSV loading, normalization, clinical masks, augmentation
  model.py                            TCN/CBAM model, RRR training wrapper, SHAP/Grad-CAM/IG
  ontology.py                         Baked OWL/RDF ontology and neuro-symbolic narratives
  pipeline.py                         End-to-end training, evaluation, XAI, reporting

webapp/
  app.py                              Streamlit dashboard
  overview_hero.html                  Overview page architecture cards
  .streamlit/config.toml              Streamlit theme and port configuration

data/
  mitbih_train.csv                    Training data
  mitbih_test.csv                     Test data

output/
  ecg_tcn_cbam_hybrid.keras           Saved model
  figures/                            Confusion matrix, clinical occlusion, training, and XAI plots
  reports/                            Classification, ontology, experiment, occlusion, training, and XAI reports
  xai/                                Per-sample XAI overlays, comparisons, and ontology reports
```

## Environment

Python 3.10+ is recommended. The current local validation used Python 3.11.9.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy==2.3.5 pandas==2.3.1 scipy==1.16.1 scikit-learn==1.7.1 matplotlib==3.10.3 seaborn==0.13.2 tensorflow==2.20.0 shap==0.50.0 rdflib==7.6.0 streamlit==1.57.0 Pillow==12.2.0 plotly==6.7.0
```

The same pinned dependency list is documented in `requirements.md`.

Git LFS is required to fetch the included dataset/model artifacts from GitHub:

```powershell
git lfs install
git lfs pull
```

## Run The Pipeline

From the repository root:

```powershell
python run_pipeline.py
```

Equivalent explicit full run:

```powershell
python run_pipeline.py --mode full
```

Resume from the checked-in model and regenerate post-training XAI outputs:

```powershell
python run_pipeline.py --mode resume_post_xai
```

Skip XAI while still running baseline evaluation and the default occlusion validation:

```powershell
python run_pipeline.py --no-xai
```

Skip the clinical occlusion validation:

```powershell
python run_pipeline.py --no-occlusion-test
```

Useful CLI overrides:

```powershell
python run_pipeline.py --epochs 30 --batch-size 64 --samples-per-class 20 --shap-background-size 400
python run_pipeline.py --data-dir C:\path\to\data --output-dir C:\path\to\output
```

CLI flags:

| Flag | Meaning | Default |
|---|---|---|
| `--data-dir` | Folder containing `mitbih_train.csv` and `mitbih_test.csv` | `data/` |
| `--medical-engine-dir` | Legacy external path option retained in config | `~/Desktop/simplified model` |
| `--output-dir` | Output artifact folder | `output/` |
| `--mode` | `full` or `resume_post_xai` | `full` |
| `--epochs` | Training epochs | `20` |
| `--batch-size` | Batch size | `64` |
| `--samples-per-class` | XAI samples per class | `20` |
| `--shap-background-size` | SHAP background examples | `200` |
| `--no-xai` | Disable XAI generation | `False` |
| `--no-occlusion-test` | Disable active-beat clinical occlusion validation | `False` |

## Run The Dashboard

```powershell
streamlit run webapp/app.py
```

The dashboard runs on port `8501` by default from `webapp/.streamlit/config.toml`.

Pages:

- `Overview`: architecture and validation summary cards.
- `Run Pipeline`: launches `run_pipeline.py` with configurable mode, epochs, batch size, XAI samples per class, SHAP background size, and XAI skip toggle.
- `XAI Explorer`: sample picker, classification card, ontology explanation, XAI overlays, method comparison, clinical alignment validation, and model performance plots.
- `Reports`: text reports and CSV tables with download buttons.

## Clinical Occlusion Validation

The pipeline can run an inference-time active-beat occlusion test after baseline evaluation. It expands sparse P/QRS/T detections into a full active-beat envelope, optionally zeros the tabular clinical features, predicts again, and writes `clinical_occlusion_summary.txt`, `clinical_occlusion_per_sample.csv`, and `clinical_occlusion_comparison.png`.

## XAI Metrics

`output/reports/xai_clinical_comparison.csv` contains one row per sample and XAI method.

Key columns:

| Column | Meaning |
|---|---|
| `sample_idx` | Test-set index |
| `method` | `SHAP`, `GradCAM`, or `IntegratedGradients` |
| `y_true`, `y_pred` | Ground-truth and predicted MIT-BIH class IDs |
| `pred_confidence` | Softmax confidence for the predicted class |
| `comparison_valid` | Whether the clinical mask comparison is usable |
| `dice` | Raw XAI-vs-clinical mask overlap |
| `alignment_score` | Edge-aware Dice score, penalized by edge focus |
| `alignment_grade` | Qualitative score bucket |
| `edge_focus` | Fraction of XAI mask in ignored boundary regions |
| `threshold`, `threshold_policy` | XAI mask binarization details |
| `xai_coverage`, `clinical_coverage`, `coverage_delta` | Coverage diagnostics |

Per-class Dice telemetry is intentionally not written; reporting is summarized globally and by XAI method.

## Ontology Reports

The baked ontology in `ecg_xai_pipeline/ontology.py` models the five MIT-BIH/AAMI-style classes used by the classifier:

- Normal Beat
- Supraventricular Ectopic Beat
- Ventricular Ectopic Beat
- Fusion Beat
- Paced / Unknown Beat

It does not directly implement a broader clinician table containing atrial fibrillation or ventricular tachycardia classes.

## GitHub Packaging

This private repository includes source code, dataset CSVs, saved model, output figures, output reports, and per-sample XAI artifacts. `.venv/`, Python caches, local environment files, editor folders, and Streamlit secrets remain ignored because they are machine-local or sensitive runtime state.
