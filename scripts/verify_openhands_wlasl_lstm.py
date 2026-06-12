#!/usr/bin/env python3

from __future__ import annotations

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
CONFIG_PATH = ROOT / "assets/openhands/wlasl_lstm/wlasl/lstm/lstm.yaml"
CHECKPOINT_PATH = ROOT / "assets/openhands/wlasl_lstm/wlasl/lstm/epoch=109-step=49059.ckpt"
SPLIT_FILE = ROOT / "assets/openhands/wlasl_metadata/splits/asl2000.json"


cfg = OmegaConf.load(str(CONFIG_PATH))
cfg.pretrained = str(CHECKPOINT_PATH)
cfg.data.test_pipeline = OmegaConf.create(
    {
        "dataset": {
            "_target_": "openhands.datasets.isolated.WLASLDataset",
            "root_dir": str(ROOT / "assets/openhands"),
            "split_file": str(SPLIT_FILE),
            "splits": "test",
            "modality": "pose",
            "only_metadata": True,
        },
        "transforms": cfg.data.valid_pipeline.transforms,
        "dataloader": cfg.data.valid_pipeline.dataloader,
    }
)

model = InferenceModel(cfg=cfg, stage="test")
model.init_from_checkpoint_if_available()
print("OpenHands WLASL LSTM checkpoint loaded successfully.")
print(f"Classes: {model.datamodule.num_class}")
print(f"Channels: {model.datamodule.in_channels}")
