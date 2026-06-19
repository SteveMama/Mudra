#!/usr/bin/env python3
"""
Mudra Live Translate — native macOS app.
OpenCV camera window + MediaPipe Python landmark detection (no browser, no WebGL).

Keys:
  q / ESC  quit
  f        toggle word-sign / fingerspell mode
  c        clear output
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms._functional_tensor as _ft

sys.modules["torchvision.transforms.functional_tensor"] = _ft

from omegaconf import OmegaConf
from openhands.datasets.pose_transforms import CenterAndScaleNormalize
from openhands.models.loader import get_model

ROOT = Path(__file__).resolve().parent

# ── paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH     = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/config.yaml"
CHECKPOINT_PATH = ROOT / "assets/openhands/wlasl_stgcn/wlasl/st_gcn/epoch=212-step=95210.ckpt"
SPLIT_FILE      = ROOT / "assets/openhands/wlasl_metadata/splits/asl2000.json"
FS_MODEL_PATH   = ROOT / "assets/fingerspell/model.tflite"

# ── 27-pt skeleton (mediapipe_holistic_minimal_27 preset) ─────────────────────
POSE_PICK = [0, 2, 5, 11, 12, 13, 14]          # nose, eye outer ×2, shoulders, elbows
HAND_PICK = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]  # wrist + fingertips + MCPs

# ── fingerspell model ─────────────────────────────────────────────────────────
FACE_IDX_76 = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308, 1, 2, 98, 327, 33, 7, 163, 144,
    145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390,
    373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
]
POSE_IDX_12 = list(range(11, 23))

_IDX_TO_CHAR = {v: k for k, v in {
    " ": 0, "!": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8,
    "*": 9, "+": 10, ",": 11, "-": 12, ".": 13, "/": 14, "0": 15, "1": 16,
    "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24,
    ":": 25, ";": 26, "=": 27, "?": 28, "@": 29, "[": 30, "_": 31,
    "a": 32, "b": 33, "c": 34, "d": 35, "e": 36, "f": 37, "g": 38, "h": 39,
    "i": 40, "j": 41, "k": 42, "l": 43, "m": 44, "n": 45, "o": 46, "p": 47,
    "q": 48, "r": 49, "s": 50, "t": 51, "u": 52, "v": 53, "w": 54, "x": 55,
    "y": 56, "z": 57, "~": 58,
}.items()}

# ── sign boundary detection ───────────────────────────────────────────────────
MOTION_PTS              = [7, 10, 12, 14, 16, 17, 20, 22, 24, 26]
HOLD_VEL_THRESH         = 0.007
HOLD_FRAMES             = 12
MIN_FRAMES              = 8
MAX_FRAMES              = 150


class SignBoundaryDetector:
    def __init__(self):
        self.state = "idle"
        self._buf27: list   = []
        self._buf_full: list = []
        self._hold = 0
        self._prev: list | None = None

    def reset(self):
        self.state = "idle"; self._buf27 = []; self._buf_full = []
        self._hold = 0; self._prev = None

    def push(self, pts27, has_hands, full_frame):
        """Returns (state, buffered, velocity, frames27|None, full_frames|None)."""
        if self._prev is None:
            self._prev = pts27
            return self.state, 0, 0.0, None, None

        vel = sum(
            ((pts27[i]["x"] - self._prev[i]["x"])**2 + (pts27[i]["y"] - self._prev[i]["y"])**2)**0.5
            for i in MOTION_PTS
        ) / len(MOTION_PTS)
        self._prev = pts27
        moving = vel > HOLD_VEL_THRESH and has_hands

        fired27 = fired_full = None
        if self.state == "idle":
            if moving:
                self.state = "signing"
                self._buf27 = [pts27]; self._buf_full = [full_frame]; self._hold = 0
        else:
            self._buf27.append(pts27); self._buf_full.append(full_frame)
            if not moving:
                self._hold += 1
                if self._hold >= HOLD_FRAMES:
                    fired27, fired_full = self._fire()
            else:
                self._hold = 0
            if len(self._buf27) >= MAX_FRAMES:
                fired27, fired_full = self._fire()

        return self.state, len(self._buf27), vel, fired27, fired_full

    def _fire(self):
        f27, ff = self._buf27[:], self._buf_full[:]
        self.state = "idle"; self._buf27 = []; self._buf_full = []; self._hold = 0
        return (f27, ff) if len(f27) >= MIN_FRAMES else (None, None)


# ── ST-GCN ────────────────────────────────────────────────────────────────────
class OpenHandsRuntime:
    def __init__(self):
        cfg = OmegaConf.load(str(CONFIG_PATH))
        with open(SPLIT_FILE) as fh:
            self.glosses = sorted(e["gloss"] for e in json.load(fh))
        self.model = get_model(cfg.model, in_channels=2, num_class=len(self.glosses))
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in ckpt["state_dict"].items()}
        missing, _ = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing[:3]}")
        self.model.eval()
        self.norm = CenterAndScaleNormalize(
            reference_points_preset="shoulder_mediapipe_holistic_minimal_27"
        )

    def predict(self, frames27):
        seq = [[[float(p["x"]), float(p["y"])] for p in fr] for fr in frames27]
        t = torch.tensor(seq, dtype=torch.float32).permute(2, 0, 1)
        data = self.norm({"frames": t})
        with torch.no_grad():
            probs = torch.softmax(self.model(data["frames"].unsqueeze(0)).cpu(), dim=-1)[0]
            score, idx = probs.max(0)
        return self.glosses[idx.item()], score.item()


# ── TFLite fingerspell ────────────────────────────────────────────────────────
class FingerspellRuntime:
    def __init__(self):
        import tensorflow as tf
        self._interp = tf.lite.Interpreter(model_path=str(FS_MODEL_PATH))
        d = self._interp.get_input_details()[0]
        self._in = d["index"]
        self._out = self._interp.get_output_details()[0]["index"]

    def predict(self, full_frames):
        T = len(full_frames)
        arrs = {k: np.zeros((T, n), np.float32)
                for k, n in [("fx",76),("fy",76),("fz",76),
                              ("lx",21),("ly",21),("lz",21),
                              ("rx",21),("ry",21),("rz",21),
                              ("px",12),("py",12),("pz",12)]}

        def fill(xk, yk, zk, pts):
            if not pts:
                return
            for j, pt in enumerate(pts):
                if j >= arrs[xk].shape[1]: break
                arrs[xk][t, j] = pt["x"]; arrs[yk][t, j] = pt["y"]; arrs[zk][t, j] = pt["z"]

        for t, fr in enumerate(full_frames):
            fill("fx","fy","fz", fr.get("face76"))
            fill("lx","ly","lz", fr.get("leftHand"))
            fill("rx","ry","rz", fr.get("rightHand"))
            fill("px","py","pz", fr.get("pose12"))

        tensor = np.concatenate([
            arrs["fx"],arrs["lx"],arrs["rx"],arrs["px"],
            arrs["fy"],arrs["ly"],arrs["ry"],arrs["py"],
            arrs["fz"],arrs["lz"],arrs["rz"],arrs["pz"],
        ], axis=1)
        self._interp.resize_tensor_input(self._in, [T, 390])
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._in, tensor)
        self._interp.invoke()
        logits = self._interp.get_tensor(self._out)
        return "".join(_IDX_TO_CHAR.get(int(i), "") for i in np.argmax(logits, axis=-1))


# ── landmark extraction helpers ───────────────────────────────────────────────
def _pt(lm) -> dict:
    return {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} if lm else {"x":0.,"y":0.,"z":0.}

def _select(lms_obj, indices):
    if lms_obj is None:
        return [{"x":0.,"y":0.,"z":0.}] * len(indices)
    lm = lms_obj.landmark
    return [_pt(lm[i]) for i in indices]

def _all21(lms_obj):
    if lms_obj is None:
        return [{"x":0.,"y":0.,"z":0.}] * 21
    return [_pt(lms_obj.landmark[i]) for i in range(21)]

def extract_27(r):
    return _select(r.pose_landmarks, POSE_PICK) + \
           _select(r.left_hand_landmarks, HAND_PICK) + \
           _select(r.right_hand_landmarks, HAND_PICK)

def extract_full(r):
    return {
        "face76":    [_pt(r.face_landmarks.landmark[i]) for i in FACE_IDX_76] if r.face_landmarks else None,
        "leftHand":  _all21(r.left_hand_landmarks),
        "rightHand": _all21(r.right_hand_landmarks),
        "pose12":    _select(r.pose_landmarks, POSE_IDX_12),
    }


# ── drawing ───────────────────────────────────────────────────────────────────
GREEN  = (50,  220,  80)
ORANGE = (30,  120, 220)   # BGR
WHITE  = (240, 240, 240)
CYAN   = (220, 200,  60)

_mp_draw = mp.solutions.drawing_utils
_mp_hol  = mp.solutions.holistic

def _draw_spec(color, thick=2, radius=4):
    return _mp_draw.DrawingSpec(color=color, thickness=thick, circle_radius=radius)

def draw_skeleton(img, results, signing):
    c = ORANGE if signing else GREEN
    dot = _draw_spec(c, 4, 4)
    line = _draw_spec(c, 2, 0)
    thin = _draw_spec((80,80,80), 1, 0)
    if results.left_hand_landmarks:
        _mp_draw.draw_landmarks(img, results.left_hand_landmarks,  _mp_hol.HAND_CONNECTIONS, dot, line)
    if results.right_hand_landmarks:
        _mp_draw.draw_landmarks(img, results.right_hand_landmarks, _mp_hol.HAND_CONNECTIONS, dot, line)
    if results.pose_landmarks:
        _mp_draw.draw_landmarks(img, results.pose_landmarks, _mp_hol.POSE_CONNECTIONS,
                                _draw_spec(c, 3, 3), thin)

def put_bg(img, text, pos, scale=0.55, fg=WHITE, bg=(20,20,20), thick=1, pad=5):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


# ── app ───────────────────────────────────────────────────────────────────────
class MudraApp:
    def __init__(self):
        self.mode         = "signs"
        self.last_result  = ""
        self.last_conf    = None
        self.phrase_tokens: list[str] = []
        self.pending      = False
        self._lock        = threading.Lock()

        print("[mudra] Loading ST-GCN…")
        self.sign_rt = OpenHandsRuntime()
        print("[mudra] ST-GCN ready.")

        self.fs_rt = None
        if FS_MODEL_PATH.exists():
            print("[mudra] Loading fingerspelling TFLite…")
            try:
                self.fs_rt = FingerspellRuntime()
                print("[mudra] Fingerspelling ready.")
            except Exception as exc:
                print(f"[mudra] Fingerspelling failed: {exc}")

        self.detector = SignBoundaryDetector()
        self.holistic  = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def _infer(self, mode, f27, ff):
        try:
            if mode == "fingerspell" and self.fs_rt:
                res = self.fs_rt.predict(ff); conf = None
            else:
                res, conf = self.sign_rt.predict(f27)
            with self._lock:
                self.last_result = res; self.last_conf = conf
                self.phrase_tokens.append(res)
                print(f"[mudra] {mode}: {res}" + (f"  {conf*100:.0f}%" if conf else ""))
        except Exception as exc:
            with self._lock:
                self.last_result = f"err: {exc}"
            print(f"[mudra] inference error: {exc}")
        finally:
            with self._lock:
                self.pending = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        WIN = "Mudra  |  f=fingerspell  c=clear  q=quit"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 760)

        if not cap.isOpened():
            blank = np.zeros((400, 700, 3), np.uint8)
            msg = [
                "Camera access denied.",
                "",
                "Fix: System Settings > Privacy & Security",
                "> Camera > enable your terminal app",
                "",
                "Then rerun:  python app.py",
                "",
                "Press any key to exit.",
            ]
            for i, line in enumerate(msg):
                cv2.putText(blank, line, (30, 50 + i * 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
            cv2.imshow(WIN, blank)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res    = self.holistic.process(rgb)
            rgb.flags.writeable = True

            has_hands = res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
            pts27     = extract_27(res)
            full      = extract_full(res)

            det_state, buffered, vel, f27, ff = self.detector.push(pts27, has_hands, full)

            with self._lock:
                cur_mode = self.mode
                pending  = self.pending

            if f27 is not None and not pending:
                with self._lock:
                    self.pending = True
                threading.Thread(target=self._infer, args=(cur_mode, f27, ff), daemon=True).start()

            # ── render ────────────────────────────────────────────────────────
            signing = det_state == "signing"
            draw_skeleton(frame, res, signing)

            h, w = frame.shape[:2]
            BAR = 95

            # bottom status bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-BAR), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            mode_c = (0, 180, 255) if cur_mode == "fingerspell" else (80, 200, 80)
            cv2.putText(frame, cur_mode.upper(), (12, h-BAR+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_c, 1, cv2.LINE_AA)

            state_lbl = "SIGNING" if signing else "watching"
            cv2.putText(frame, f"{state_lbl}  |  motion {vel*1000:.1f}  |  buf {buffered}",
                        (12, h-BAR+44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

            # velocity bar
            bw = w - 24
            cv2.rectangle(frame, (12, h-BAR+52), (w-12, h-BAR+62), (55,55,55), -1)
            fill = int(min(vel / 0.05, 1.0) * bw)
            if fill > 0:
                cv2.rectangle(frame, (12, h-BAR+52), (12+fill, h-BAR+62),
                              ORANGE if signing else GREEN, -1)
            tick = 12 + int(0.007/0.05 * bw)
            cv2.line(frame, (tick, h-BAR+48), (tick, h-BAR+66), (220,220,220), 2)

            with self._lock:
                result = self.last_result
                conf   = self.last_conf
                phrase = " ".join(self.phrase_tokens[-8:])

            if result:
                conf_s = f"  {conf*100:.0f}%" if conf is not None else ""
                cv2.putText(frame, f"→ {result}{conf_s}", (12, h-BAR+82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1, cv2.LINE_AA)

            # phrase at top
            if phrase:
                put_bg(frame, phrase, (12, 40), scale=0.85,
                       fg=WHITE, bg=(0,0,0), thick=2, pad=8)

            # hands badge
            if has_hands:
                n = sum([res.left_hand_landmarks is not None, res.right_hand_landmarks is not None])
                bc = ORANGE if signing else GREEN
                put_bg(frame, f"{n} hand{'s' if n>1 else ''} detected",
                       (12, 70), scale=0.45, fg=(0,0,0), bg=bc, pad=5)
            else:
                put_bg(frame, "no hands — show your hand(s)",
                       (12, 70), scale=0.45, fg=WHITE, bg=(60,60,60), pad=5)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('f'):
                with self._lock:
                    self.mode = "fingerspell" if self.mode == "signs" else "signs"
                self.detector.reset()
                print(f"[mudra] mode → {self.mode}")
            elif key == ord('c'):
                with self._lock:
                    self.phrase_tokens = []; self.last_result = ""; self.last_conf = None
                self.detector.reset()
                print("[mudra] cleared")

        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


if __name__ == "__main__":
    MudraApp().run()
