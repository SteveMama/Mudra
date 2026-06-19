#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np
import torch
import torchvision.transforms._functional_tensor as _functional_tensor
from groq import Groq
from omegaconf import OmegaConf

sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from openhands.datasets.pose_transforms import CenterAndScaleNormalize
from openhands.models.loader import get_model


ROOT = Path(__file__).resolve().parent

# ── ST-GCN word-sign model paths ─────────────────────────────────────────────
CONFIG_PATH = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/config.yaml"
CHECKPOINT_PATH = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/epoch=212-step=95210.ckpt"
SPLIT_FILE = ROOT / "assets/openhands/wlasl_metadata/splits/asl2000.json"

# ── Kaggle fingerspelling model paths ─────────────────────────────────────────
FS_MODEL_PATH = ROOT / "assets/fingerspell/model.tflite"

# Face landmark indices from inference_args.json (76 MediaPipe face mesh indices, in order)
FACE_INDICES = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308, 1, 2, 98, 327, 33, 7, 163, 144,
    145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390,
    373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
]  # len == 76

# Pose landmark indices (MediaPipe pose 11-22 = shoulders, elbows, wrists, hips, knees, ankles)
POSE_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # len == 12

# Character vocabulary (from competition's character_to_prediction_index.json)
_CHAR_TO_IDX: dict[str, int] = {
    " ": 0, "!": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8,
    "*": 9, "+": 10, ",": 11, "-": 12, ".": 13, "/": 14, "0": 15, "1": 16,
    "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24,
    ":": 25, ";": 26, "=": 27, "?": 28, "@": 29, "[": 30, "_": 31,
    "a": 32, "b": 33, "c": 34, "d": 35, "e": 36, "f": 37, "g": 38, "h": 39,
    "i": 40, "j": 41, "k": 42, "l": 43, "m": 44, "n": 45, "o": 46, "p": 47,
    "q": 48, "r": 49, "s": 50, "t": 51, "u": 52, "v": 53, "w": 54, "x": 55,
    "y": 56, "z": 57, "~": 58,
    # 60=PAD, 61=SOS, 62=EOS  — all decode to ""
}
_IDX_TO_CHAR: dict[int, str] = {v: k for k, v in _CHAR_TO_IDX.items()}

# ── Language names for Groq prompt ───────────────────────────────────────────
LANGUAGE_NAMES: dict[str, str] = {
  "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
  "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
  "pt": "Portuguese", "ar": "Arabic", "zh": "Mandarin Chinese",
  "ja": "Japanese", "ko": "Korean", "ru": "Russian", "tr": "Turkish",
  "vi": "Vietnamese", "id": "Indonesian", "sw": "Swahili",
}

# ── Groq translation ──────────────────────────────────────────────────────────

class GroqTranslator:
  def __init__(self) -> None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
      raise RuntimeError("GROQ_API_KEY environment variable not set.")
    self._client = Groq(api_key=api_key)

  def translate(self, text: str, target_lang: str) -> str:
    if target_lang == "en":
      return text
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    resp = self._client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[
        {
          "role": "system",
          "content": (
            f"You are a translator. Translate the user's text into {lang_name}. "
            "Return only the translated text, no explanation, no quotes."
          ),
        },
        {"role": "user", "content": text},
      ],
      temperature=0.2,
      max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# ── OpenHands ST-GCN (word signs) ─────────────────────────────────────────────

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
    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
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

  def _normalize_state_dict(self, state_dict: dict) -> dict:
    return {
      (key[len("model."):] if key.startswith("model.") else key): value
      for key, value in state_dict.items()
    }

  def predict(self, frames: list) -> dict:
    if len(frames) < 8:
      raise ValueError("At least 8 frames required.")
    for frame in frames:
      if len(frame) != 27:
        raise ValueError("Each frame must contain exactly 27 points.")

    sequence = [
      [[float(p.get("x", 0)), float(p.get("y", 0))] for p in frame]
      for frame in frames
    ]
    tensor = torch.tensor(sequence, dtype=torch.float32).permute(2, 0, 1)
    data = self.normalizer({"frames": tensor})
    input_tensor = data["frames"].unsqueeze(0)

    with torch.no_grad():
      logits = self.model(input_tensor).cpu()
      probs = torch.softmax(logits, dim=-1)[0]
      values, indices = torch.topk(probs, k=5)

    predictions = [
      {"gloss": self.glosses[idx], "score": score}
      for score, idx in zip(values.tolist(), indices.tolist())
    ]
    return {"topPrediction": predictions[0], "predictions": predictions, "framesUsed": len(frames)}


# ── Kaggle fingerspelling (Squeezeformer TFLite) ──────────────────────────────

class FingerspellRuntime:
  def __init__(self) -> None:
    import tensorflow as tf
    self._interp = tf.lite.Interpreter(model_path=str(FS_MODEL_PATH))
    self._input_idx = self._interp.get_input_details()[0]["index"]
    self._output_idx = self._interp.get_output_details()[0]["index"]

  def predict(self, frames: list) -> dict:
    """
    frames: list of dicts, each with keys:
      face76  – list of 76 {x,y,z} dicts (in FACE_INDICES order) or null/missing
      leftHand  – list of 21 {x,y,z} dicts
      rightHand – list of 21 {x,y,z} dicts
      pose12  – list of 12 {x,y,z} dicts (pose landmarks 11-22)
    """
    if len(frames) < 15:
      raise ValueError("At least 15 frames required for fingerspelling.")

    T = len(frames)
    # Build arrays for each body part
    face_x = np.zeros((T, 76), np.float32)
    face_y = np.zeros((T, 76), np.float32)
    face_z = np.zeros((T, 76), np.float32)
    lh_x = np.zeros((T, 21), np.float32)
    lh_y = np.zeros((T, 21), np.float32)
    lh_z = np.zeros((T, 21), np.float32)
    rh_x = np.zeros((T, 21), np.float32)
    rh_y = np.zeros((T, 21), np.float32)
    rh_z = np.zeros((T, 21), np.float32)
    po_x = np.zeros((T, 12), np.float32)
    po_y = np.zeros((T, 12), np.float32)
    po_z = np.zeros((T, 12), np.float32)

    def fill(arr_x, arr_y, arr_z, pts, t):
      for j, pt in enumerate(pts or []):
        if j >= arr_x.shape[1]:
          break
        arr_x[t, j] = float(pt.get("x", 0))
        arr_y[t, j] = float(pt.get("y", 0))
        arr_z[t, j] = float(pt.get("z", 0))

    for t, frame in enumerate(frames):
      fill(face_x, face_y, face_z, frame.get("face76"), t)
      fill(lh_x,   lh_y,   lh_z,   frame.get("leftHand"), t)
      fill(rh_x,   rh_y,   rh_z,   frame.get("rightHand"), t)
      fill(po_x,   po_y,   po_z,   frame.get("pose12"), t)

    # Column order: x_face(76) x_lhand(21) x_rhand(21) x_pose(12)  [then y_, then z_]
    tensor = np.concatenate([
      face_x, lh_x, rh_x, po_x,
      face_y, lh_y, rh_y, po_y,
      face_z, lh_z, rh_z, po_z,
    ], axis=1)  # (T, 390)

    self._interp.resize_tensor_input(self._input_idx, [T, 390])
    self._interp.allocate_tensors()
    self._interp.set_tensor(self._input_idx, tensor)
    self._interp.invoke()

    logits = self._interp.get_tensor(self._output_idx)  # (N_chars, 63)
    char_indices = np.argmax(logits, axis=-1)
    text = "".join(_IDX_TO_CHAR.get(int(i), "") for i in char_indices)
    return {"text": text, "framesUsed": T}


# ── Boot runtimes ─────────────────────────────────────────────────────────────

print("[mudra] Loading ST-GCN word model…", flush=True)
SIGN_RUNTIME = OpenHandsRuntime()
print("[mudra] ST-GCN ready.", flush=True)

FS_RUNTIME: FingerspellRuntime | None = None
if FS_MODEL_PATH.exists():
  print("[mudra] Loading fingerspelling model…", flush=True)
  try:
    FS_RUNTIME = FingerspellRuntime()
    print("[mudra] Fingerspelling model ready.", flush=True)
  except Exception as exc:
    print(f"[mudra] Fingerspelling model failed to load: {exc}", flush=True)
else:
  print("[mudra] Fingerspelling model not found — /api/fingerspell disabled.", flush=True)

TRANSLATOR: GroqTranslator | None = None
try:
  TRANSLATOR = GroqTranslator()
  print("[mudra] Groq translator ready.", flush=True)
except RuntimeError as exc:
  print(f"[mudra] {exc} — /api/translate disabled.", flush=True)


# ── HTTP handler ──────────────────────────────────────────────────────────────

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
    if self.path == "/api/infer":
      self._handle(self._infer)
    elif self.path == "/api/fingerspell":
      self._handle(self._fingerspell)
    elif self.path == "/api/translate":
      self._handle(self._translate)
    else:
      self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

  def _handle(self, fn):
    try:
      length = int(self.headers.get("Content-Length", "0"))
      payload = json.loads(self.rfile.read(length))
      result = fn(payload)
      self._write_json(HTTPStatus.OK, result)
    except ValueError as err:
      self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
    except Exception as err:  # noqa: BLE001
      self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(err)})

  def _infer(self, payload: dict) -> dict:
    return SIGN_RUNTIME.predict(payload.get("frames", []))

  def _fingerspell(self, payload: dict) -> dict:
    if FS_RUNTIME is None:
      raise ValueError("Fingerspelling model not available.")
    return FS_RUNTIME.predict(payload.get("frames", []))

  def _translate(self, payload: dict) -> dict:
    if TRANSLATOR is None:
      raise ValueError("Translator not available — set GROQ_API_KEY.")
    text = payload.get("text", "").strip()
    lang = payload.get("lang", "en")
    if not text:
      raise ValueError("text is required.")
    translated = TRANSLATOR.translate(text, lang)
    return {"translated": translated, "lang": lang}

  def _write_json(self, status: HTTPStatus, data: dict):
    body = json.dumps(data).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def main():
  host = os.environ.get("HOST", "127.0.0.1")
  port = int(os.environ.get("PORT", "4173"))
  server = ThreadingHTTPServer((host, port), MudraHandler)
  print(f"[mudra] Server running at http://{host}:{port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
