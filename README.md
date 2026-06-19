# Mudra Live Translate

Real-time ASL-to-text in the browser. Sign a word, hold still, get an English gloss and an instant translation into 18 languages — powered by a local ST-GCN model and Groq's free LLM API.

> **macOS + Safari only** for now. Chrome's WebGL is disabled on some Macs; Safari works reliably.

---

## Quick start

```bash
git clone https://github.com/SteveMama/Mudra.git
cd Mudra
bash setup.sh
```

`setup.sh` does everything in one shot:

- creates a Python venv (`.venv-mudra`)
- installs all Python packages
- installs npm packages for the browser frontend
- downloads the 4 MediaPipe landmark models
- downloads the Kaggle ASL fingerspelling model (39 MB)
- prompts for your Groq API key and saves it to `.env`

Then run the server:

```bash
.venv-mudra/bin/python server.py
```

Open **Safari** at `http://127.0.0.1:4173`, click **Start camera**, and sign.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| macOS | Required for Safari camera access |
| Python 3.9+ | `brew install python` if needed |
| Node.js + npm | For the browser frontend JS — `brew install node` |
| Groq API key | Free at [console.groq.com](https://console.groq.com) |

---

## How it works

### Sign detection (browser)

1. **MediaPipe** runs in Safari — hand, pose, and face landmarks extracted every frame on CPU
2. **Sign boundary detector** watches wrist velocity — orange border fires when you hold a sign still for ~0.5 s
3. Frames are sent to the local Python server for inference

### Inference (local Python server)

Two modes, toggled by the dropdown:

| Mode | Model | What it recognises |
|---|---|---|
| Word signs | ST-GCN (OpenHands / WLASL 2000) | 2,000 ASL words |
| Fingerspelling | Squeezeformer TFLite (Kaggle 1st place) | A–Z letter sequences |

### Translation (Groq cloud)

Detected English text is sent to **Groq** (`llama-3.1-8b-instant`) and translated in real time. The translation panel supports 18 languages.

---

## API endpoints

All served by `server.py` on `http://127.0.0.1:4173`.

### `POST /api/infer`
Word-sign inference (ST-GCN).

```json
// request
{ "frames": [ [{"x": 0.1, "y": 0.2}, ...] ] }   // 27 points × ≥8 frames

// response
{ "topPrediction": {"gloss": "hello", "score": 0.82}, "predictions": [...], "framesUsed": 36 }
```

### `POST /api/fingerspell`
Fingerspelling inference (TFLite Squeezeformer).

```json
// request
{ "frames": [ {"face76": [...], "leftHand": [...], "rightHand": [...], "pose12": [...]} ] }

// response
{ "text": "hello", "framesUsed": 42 }
```

### `POST /api/translate`
Groq-powered translation.

```json
// request
{ "text": "hello", "lang": "hi" }

// response
{ "translated": "नमस्ते", "lang": "hi" }
```

---

## Supported languages

English · Hindi · Telugu · Tamil · Spanish · French · German · Italian · Portuguese · Arabic · Chinese · Japanese · Korean · Russian · Turkish · Vietnamese · Indonesian · Swahili

---

## Project structure

```
Mudra/
├── setup.sh                  # one-shot setup script
├── server.py                 # Python server — inference + translation + static files
├── app.py                    # native macOS OpenCV app (alternative to browser)
├── requirements.txt          # Python dependencies
├── index.html                # browser UI
├── src/
│   ├── main.js               # app state, sign callback, phrase accumulation
│   ├── visionController.js   # MediaPipe landmark extraction
│   ├── modelAdapter.js       # sign boundary detector
│   ├── translationService.js # calls /api/translate
│   ├── demoPhrases.js        # language list
│   └── styles.css
├── models/                   # MediaPipe .task files (downloaded by setup.sh)
├── assets/
│   ├── openhands/            # ST-GCN checkpoint + WLASL metadata (in repo)
│   └── fingerspell/          # TFLite model (downloaded by setup.sh)
└── .env                      # GROQ_API_KEY (gitignored, created by setup.sh)
```

---

## Try these signs first (Word signs mode)

The ST-GCN model knows 2,000 ASL words. Good ones to start with:

`hello` · `yes` · `no` · `thank you` · `help` · `please` · `sorry` · `good` · `bad` · `water` · `food` · `home` · `work` · `school` · `friend` · `family` · `love` · `name`

**How to trigger:**
1. Show your hand — green skeleton should appear immediately
2. Make the sign with a clear motion (velocity bar fills past the white tick)
3. **Hold still** for ~0.5 s → orange border fires → result appears

---

## Limitations

- **Accuracy**: The WLASL ST-GCN checkpoint was trained on benchmark video, not your webcam/background/style. Expect ~30–50% top-1 accuracy on clean signs.
- **Isolated words only**: Word signs mode recognises one sign at a time, not continuous sentences.
- **macOS + Safari**: Chrome's WebGL fails on some machines. Safari required for browser mode.
- **CPU inference**: No GPU acceleration — ~50–100 ms per sign.

---

## Sources

- [AI4Bharat OpenHands](https://github.com/AI4Bharat/OpenHands)
- [WLASL dataset](https://dxli94.github.io/WLASL/)
- [Kaggle ASL Fingerspelling — 1st place solution](https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution)
- [Groq](https://console.groq.com)
