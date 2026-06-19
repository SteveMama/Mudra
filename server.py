#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torchvision.transforms._functional_tensor as _functional_tensor
from omegaconf import OmegaConf

sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from openhands.datasets.pose_transforms import CenterAndScaleNormalize
from openhands.models.loader import get_model


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/config.yaml"
CHECKPOINT_PATH = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/epoch=212-step=95210.ckpt"
SPLIT_FILE = ROOT / "assets/openhands/wlasl_metadata/splits/asl2000.json"


class OpenHandsRuntime:
  def __init__(self) -> None:
    self.cfg = OmegaConf.load(str(CONFIG_PATH))
    self.glosses = self._load_glosses()
    self.model = get_model(self.cfg.model, in_channels=2, num_class=len(self.glosses))
    checkpoint = torch.load(
      str(CHECKPOINT_PATH),
      map_location=torch.device("cpu"),
      weights_only=False,
    )
    state_dict = self._normalize_state_dict(checkpoint["state_dict"])
    missing_keys, unexpected_keys = self.model.load_state_dict(
      state_dict,
      strict=False,
    )
    if missing_keys:
      raise RuntimeError(
        f"OpenHands checkpoint is missing {len(missing_keys)} model keys — "
        f"architecture mismatch: {missing_keys[:5]}"
      )
    if unexpected_keys:
      print(
        f"[mudra] checkpoint has {len(unexpected_keys)} extra keys (Lightning artefacts, ignored)",
        flush=True,
      )
    self.model.eval()
    self.normalizer = CenterAndScaleNormalize(
      reference_points_preset="shoulder_mediapipe_holistic_minimal_27"
    )

  def _load_glosses(self) -> list[str]:
    with open(SPLIT_FILE, "r", encoding="utf-8") as handle:
      content = json.load(handle)
    return sorted(entry["gloss"] for entry in content)

  def _normalize_state_dict(self, state_dict: dict[str, object]) -> dict[str, object]:
    normalized = {}
    for key, value in state_dict.items():
      if key.startswith("model."):
        normalized[key[len("model."):]] = value
      else:
        normalized[key] = value
    return normalized

  def predict(self, frames: list[list[dict[str, float]]]) -> dict[str, object]:
    if not frames:
      raise ValueError("No frames provided.")
    if len(frames) < 8:
      raise ValueError("At least 8 frames are required for a stable prediction.")

    sequence = []
    for frame in frames:
      if len(frame) != 27:
        raise ValueError("Each frame must contain exactly 27 points.")
      sequence.append(
        [[float(point.get("x", 0.0)), float(point.get("y", 0.0))] for point in frame]
      )

    tensor = torch.tensor(sequence, dtype=torch.float32).permute(2, 0, 1)
    data = {"frames": tensor}
    data = self.normalizer(data)
    input_tensor = data["frames"].unsqueeze(0)

    with torch.no_grad():
      logits = self.model(input_tensor).cpu()
      probabilities = torch.softmax(logits, dim=-1)[0]
      values, indices = torch.topk(probabilities, k=5)

    predictions = []
    for score, index in zip(values.tolist(), indices.tolist()):
      predictions.append(
        {
          "gloss": self.glosses[index],
          "score": score,
        }
      )

    return {
      "topPrediction": predictions[0],
      "predictions": predictions,
      "framesUsed": len(frames),
    }


RUNTIME = OpenHandsRuntime()


class MudraHandler(SimpleHTTPRequestHandler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(ROOT), **kwargs)

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    self.send_header("Access-Control-Allow-Origin", "*")
    super().end_headers()

  def do_OPTIONS(self):
    self.send_response(200)
    self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")
    self.end_headers()

  def do_POST(self):
    if self.path != "/api/infer":
      self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
      return

    try:
      content_length = int(self.headers.get("Content-Length", "0"))
      payload = json.loads(self.rfile.read(content_length))
      result = RUNTIME.predict(payload.get("frames", []))
      body = json.dumps(result).encode("utf-8")
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    except ValueError as error:
      self._write_error(HTTPStatus.BAD_REQUEST, str(error))
    except Exception as error:  # noqa: BLE001
      self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

  def _write_error(self, status: HTTPStatus, message: str):
    body = json.dumps({"error": message}).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def main():
  host = os.environ.get("HOST", "127.0.0.1")
  port = int(os.environ.get("PORT", "4173"))
  server = ThreadingHTTPServer((host, port), MudraHandler)
  print(f"Mudra server running at http://{host}:{port}")
  server.serve_forever()


if __name__ == "__main__":
  main()
