# =============================================================================
# IMPORTS
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EXTERNAL_ROOT = Path.home() / "Desktop" / "simplified model"


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for reproducible experiments."""

    data_dir: Path = Path(os.getenv("MITBIH_DATA_DIR", DEFAULT_DATA_DIR))
    medical_engine_dir: Path = Path(
        os.getenv("MEDICAL_ENGINE_DIR", DEFAULT_EXTERNAL_ROOT)
    )
    output_dir: Path = PROJECT_ROOT / "output"

    train_csv: str = "mitbih_train.csv"
    test_csv: str = "mitbih_test.csv"
    model_name: str = "ecg_tcn_cbam_hybrid.keras"

    sampling_rate: int = 125
    num_classes: int = 5
    num_features: int = 3
    crop_start: int = 20
    crop_end: int = 167

    validation_size: float = 0.15
    random_seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3

    clinical_mask_fraction: float = 0.30
    modality_dropout_rate: float = 0.20
    tukey_alpha: float = 0.10
    max_shift: int = 15
    # Per-class oversampling factors for minority augmentation.
    # Keys = class labels, values = total copies (including original).
    augmentation_factors: Tuple[Tuple[int, int], ...] = (
        (1, 5), (3, 5), (2, 2), (4, 2)
    )

    shap_background_size: int = 200
    shap_use_aux_head: bool = True
    integrated_gradients_steps: int = 64
    integrated_gradients_use_aux_head: bool = True
    integrated_gradients_smooth_samples: int = 4
    integrated_gradients_noise_std: float = 0.01
    gradcam_layer_name: str = "stem2_relu"
    gradcam_use_aux_head: bool = True
    xai_score_mode: str = "log_probability"
    samples_per_class: int = 20
    xai_percentile: float = 75.0
    xai_match_clinical_coverage: bool = True
    xai_smoothing_sigma: float = 3.5
    xai_zero_edges_before_threshold: bool = True
    dilation_radius: int = 2
    ignore_edge: int = 15

    rrr_lambda: float = 0.05
    rrr_warmup_epochs: int = 3
    rrr_use_aux_head: bool = True

    mode: str = "full"
    run_xai: bool = True
    run_clinical_occlusion_test: bool = True
    clinical_occlusion_zero_features: bool = True
    # "pqrst" = sparse P/QRS/T islands only; "active_beat" = full beat envelope
    clinical_occlusion_mode: str = "active_beat"
    clinical_occlusion_dilation_radius: int = 35

    @property
    def cropped_len(self) -> int:
        return self.crop_end - self.crop_start

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_csv

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_csv

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_name

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def xai_dir(self) -> Path:
        return self.output_dir / "xai"

    @property
    def medical_dir(self) -> Path:
        return self.output_dir / "medical"

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.output_dir,
            self.reports_dir,
            self.figures_dir,
            self.xai_dir,
            self.medical_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
