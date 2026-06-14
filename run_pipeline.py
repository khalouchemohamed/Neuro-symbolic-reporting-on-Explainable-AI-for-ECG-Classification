from __future__ import annotations

import argparse
from pathlib import Path

from ecg_xai_pipeline.config import PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and explain a clinically guided MIT-BIH ECG classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--medical-engine-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["full", "resume_post_xai"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--shap-background-size", type=int, default=None)
    parser.add_argument("--no-xai", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = PipelineConfig()
    config = PipelineConfig(
        data_dir=args.data_dir if args.data_dir is not None else base.data_dir,
        medical_engine_dir=args.medical_engine_dir if args.medical_engine_dir is not None else base.medical_engine_dir,
        output_dir=args.output_dir if args.output_dir is not None else base.output_dir,
        mode=args.mode,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        samples_per_class=args.samples_per_class if args.samples_per_class is not None else base.samples_per_class,
        shap_background_size=args.shap_background_size if args.shap_background_size is not None else base.shap_background_size,
        run_xai=not args.no_xai,
    )

    from ecg_xai_pipeline.pipeline import run_pipeline

    run_pipeline(config)


if __name__ == "__main__":
    main()
