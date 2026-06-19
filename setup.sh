#!/usr/bin/env bash
# Mudra Live Translate — one-shot setup
# Usage: bash setup.sh
set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1;34m'; N='\033[0m'
log()  { echo -e "${G}✔${N}  $1"; }
info() { echo -e "${B}▸${N}  $1"; }
warn() { echo -e "${Y}⚠${N}  $1"; }
err()  { echo -e "${R}✘${N}  $1"; exit 1; }
hr()   { echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

hr
echo -e "   ${B}Mudra Live Translate — setup${N}"
hr
echo ""

# ── 1. System checks ──────────────────────────────────────────────────────────
info "Checking system requirements..."

[[ "$(uname)" == "Darwin" ]] || err "macOS required (open Safari for camera access)."

if ! command -v python3 &>/dev/null; then
  err "Python 3 not found. Install from https://python.org (3.9+ required)."
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
[[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 9 ]] || err "Python 3.9+ required (found $PY_VER)."
log "Python $PY_VER"

command -v curl &>/dev/null || err "curl not found."
log "curl available"

# ── 2. Python venv ────────────────────────────────────────────────────────────
VENV="$REPO/.venv-mudra"
if [ ! -d "$VENV" ]; then
  info "Creating Python virtual environment..."
  python3 -m venv "$VENV"
  log "Venv created at .venv-mudra"
else
  log "Venv already exists"
fi

PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── 3. Python packages ────────────────────────────────────────────────────────
info "Installing Python packages (this will take a few minutes on first run)..."
$PIP install --upgrade pip --quiet
$PIP install -r "$REPO/requirements.txt" --quiet
log "Python packages installed"

# ── 4. Node modules (browser frontend) ───────────────────────────────────────
if [ -f "$REPO/package.json" ]; then
  if [ ! -d "$REPO/node_modules" ]; then
    if command -v npm &>/dev/null; then
      info "Installing npm packages..."
      npm install --prefix "$REPO" --silent
      log "npm packages installed"
    else
      warn "npm not found — MediaPipe JS may not work in the browser. Install Node.js from https://nodejs.org"
    fi
  else
    log "node_modules already present"
  fi
fi

# ── 5. Download models ────────────────────────────────────────────────────────
info "Checking model files..."
mkdir -p "$REPO/models" "$REPO/assets/fingerspell"

MP="https://storage.googleapis.com/mediapipe-models"

dl() {
  local url="$1" dest="$2" label="$3"
  if [ -f "$dest" ]; then
    log "$label already downloaded"
  else
    info "Downloading $label..."
    curl -L --progress-bar -o "$dest" "$url"
    log "$label downloaded"
  fi
}

dl "$MP/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
   "$REPO/models/hand_landmarker.task" "Hand landmarker"

dl "$MP/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
   "$REPO/models/pose_landmarker_lite.task" "Pose landmarker"

dl "$MP/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task" \
   "$REPO/models/gesture_recognizer.task" "Gesture recognizer"

dl "$MP/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
   "$REPO/models/face_landmarker.task" "Face landmarker"

# Kaggle ASL fingerspelling model (39 MB zip)
if [ ! -f "$REPO/assets/fingerspell/model.tflite" ]; then
  info "Downloading Kaggle ASL fingerspelling model (39 MB)..."
  TMP_ZIP=$(mktemp /tmp/mudra_fs_XXXXXX.zip)
  curl -L --progress-bar -o "$TMP_ZIP" \
    "https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/releases/download/v0.0.1-alpha/weights.zip"
  info "Extracting fingerspelling model..."
  TMP_DIR=$(mktemp -d /tmp/mudra_fs_XXXXXX)
  unzip -q "$TMP_ZIP" -d "$TMP_DIR"
  # Find and copy the .tflite file
  TFLITE=$(find "$TMP_DIR" -name "*.tflite" | head -1)
  [ -n "$TFLITE" ] || err "Could not find .tflite in the downloaded zip."
  cp "$TFLITE" "$REPO/assets/fingerspell/model.tflite"
  # Copy inference_args.json if present
  ARGS=$(find "$TMP_DIR" -name "inference_args.json" | head -1)
  [ -n "$ARGS" ] && cp "$ARGS" "$REPO/assets/fingerspell/inference_args.json"
  rm -rf "$TMP_ZIP" "$TMP_DIR"
  log "Fingerspelling model ready"
else
  log "Fingerspelling model already downloaded"
fi

# ── 6. .env / Groq API key ────────────────────────────────────────────────────
if [ ! -f "$REPO/.env" ]; then
  echo ""
  echo -e "  ${B}Groq API key setup${N}"
  echo "  Get a free key at: https://console.groq.com"
  echo ""
  printf "  Enter your GROQ_API_KEY (or press Enter to skip): "
  read -r GROQ_KEY
  if [ -n "$GROQ_KEY" ]; then
    echo "GROQ_API_KEY=$GROQ_KEY" > "$REPO/.env"
    log "API key saved to .env"
  else
    warn "Skipped — translation will be disabled until you add GROQ_API_KEY to .env"
    echo "GROQ_API_KEY=" > "$REPO/.env"
  fi
else
  log ".env already exists"
fi

# ── 7. Quick smoke test ───────────────────────────────────────────────────────
info "Verifying install..."
$PYTHON -c "import torch, mediapipe, cv2, tensorflow, groq; print('all imports ok')" \
  && log "All Python imports verified" \
  || warn "Some packages may not have installed correctly — check output above."

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
hr
echo -e "   ${G}Setup complete!${N}"
hr
echo ""
echo -e "  ${B}To run Mudra:${N}"
echo ""
echo "    $VENV/bin/python server.py"
echo ""
echo -e "  Then open ${B}Safari${N} at:  http://127.0.0.1:4173"
echo ""
echo -e "  ${Y}Note:${N} Use Safari — Chrome's WebGL is disabled on some Macs."
echo ""
