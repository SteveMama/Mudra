#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as _functional_tensor

    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from openhands.apis.inference import InferenceModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "assets/openhands/wlasl_lstm/wlasl/lstm/lstm.yaml"
DEFAULT_CHECKPOINT = ROOT / "assets/openhands/wlasl_lstm/wlasl/lstm/epoch=109-step=49059.ckpt"
DEFAULT_SPLIT_FILE = ROOT / "assets/openhands/wlasl_metadata/splits/asl2000.json"


def build_cfg(pose_dir: Path, split_file: Path, checkpoint: Path):
    cfg = OmegaConf.load(str(DEFAULT_CONFIG))
    cfg.pretrained = str(checkpoint)

    dataset_cfg = OmegaConf.create(
        {
            "_target_": "openhands.datasets.isolated.WLASLDataset",
            "root_dir": str(pose_dir),
            "split_file": str(split_file),
            "splits": "test",
            "modality": "pose",
            "inference_mode": True,
        }
    )

    cfg.data.test_pipeline = OmegaConf.create(
        {
            "dataset": dataset_cfg,
            "transforms": cfg.data.valid_pipeline.transforms,
            "dataloader": cfg.data.valid_pipeline.dataloader,
        }
    )
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the downloaded OpenHands WLASL LSTM checkpoint on pose pickle files.",
    )
    parser.add_argument(
        "pose_dir",
        type=Path,
        help="Directory containing compatible pose .pkl files.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="WLASL metadata split JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="OpenHands checkpoint path.",
    )
    args = parser.parse_args()

    if not args.pose_dir.exists():
        raise SystemExit(f"Pose directory does not exist: {args.pose_dir}")

    cfg = build_cfg(args.pose_dir, args.split_file, args.checkpoint)
    model = InferenceModel(cfg=cfg, stage="test")
    model.init_from_checkpoint_if_available()
    model.test_inference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
