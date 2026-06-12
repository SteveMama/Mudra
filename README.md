# Mudra Live Translate

Mudra Live Translate is a local prototype for sign-to-text communication.

The app currently combines:

- a browser UI for camera capture
- MediaPipe hand and pose landmark extraction in the browser
- a local Python inference server
- a downloaded `OpenHands` `WLASL LSTM` checkpoint for isolated-word ASL gloss prediction
- a simple translation layer for English, Hindi, Telugu, Spanish, French, and Arabic

This is an honest prototype, not a full sign-language translator.

## Current state

What the app does today:

- starts the webcam in the browser
- records a short signing clip when you press `Record sign`
- extracts an OpenHands-compatible 27-point pose sequence from each frame
- sends the recorded sequence to `/api/infer`
- runs the local OpenHands `WLASL` model on CPU
- returns the top predicted English gloss
- shows a translated output in the selected language
- lets you play the translated text with browser speech synthesis

What it does not do today:

- continuous sentence translation
- full ASL conversation understanding
- reliable live webcam recognition for arbitrary signing
- fingerspelling recognition
- cloud translation or LLM-backed translation
- user adaptation or calibration

## Important limitation

The downloaded model is an **isolated-word ASL recognition** checkpoint, not a sentence-level translation model.

That means the current app is only attempting:

`single signed word -> English gloss -> translated display text`

It is **not** doing:

`free-form ASL sentence -> fluent English sentence -> multilingual translation`

Accuracy is also limited because the public `WLASL` checkpoint was trained on benchmark data, not on your specific webcam setup, background, framing, or signing style.

## Model in use

The active model path is:

- project: `AI4Bharat/OpenHands`
- dataset family: `WLASL`
- architecture: `ST-GCN`
- checkpoint: `assets/openhands/wlasl_stgcn/wlasl/st_gcn/epoch=212-step=95210.ckpt`

The local server loads:

- config: `assets/openhands/wlasl_stgcn/wlasl/st_gcn/config.yaml`
- metadata: `assets/openhands/wlasl_metadata/splits/asl2000.json`

The backend applies OpenHands pose normalization before inference and returns the top 5 gloss predictions.

## How the app works

### 1. Browser capture

The frontend uses:

- `getUserMedia` for webcam access
- `@mediapipe/tasks-vision`
- `HandLandmarker`
- `PoseLandmarker`

The browser collects:

- 7 selected pose landmarks
- 10 selected left-hand landmarks
- 10 selected right-hand landmarks

These are assembled into a 27-point frame sequence compatible with the current OpenHands integration.

### 2. Local inference server

`server.py` serves both:

- the static frontend
- the inference API at `POST /api/infer`

The server:

- validates the incoming frame sequence
- requires at least 8 frames
- converts frames into a tensor
- normalizes the pose sequence
- runs the OpenHands checkpoint on CPU
- returns top-k gloss predictions and scores

### 3. Translation layer

The translation layer is currently minimal.

It only has explicit translations for the demo phrases in `src/demoPhrases.js`. For any model prediction that is not one of those phrases, the translated output currently falls back to the English text unchanged.

That means:

- demo phrases translate properly
- model-predicted glosses usually appear unchanged in non-English languages

## Current UI behavior

The main UI supports:

- `Start camera`
- `Stop camera`
- `Record sign`
- `Clear`
- target language selection
- `Speak output`
- demo phrase playback

During recording:

- the app draws a simple landmark overlay
- buffers captured frames
- shows hand count and buffered frame count

When recording stops:

- the app calls the local inference API
- displays the top English prediction
- shows the translated output
- appends the result to the session log

There is also still a demo phrase section in the UI. That path is manual and separate from real model inference.

## Project structure

Key files:

- `index.html`
  - main UI shell
- `src/main.js`
  - browser state, recording flow, API calls, translation rendering, speech output
- `src/visionController.js`
  - MediaPipe setup and 27-point feature extraction
- `src/modelAdapter.js`
  - recorded sequence buffer
- `src/translationService.js`
  - simple translation lookup and fallback behavior
- `src/demoPhrases.js`
  - demo phrases and supported language labels
- `src/styles.css`
  - styling
- `server.py`
  - static server plus OpenHands inference endpoint
- `scripts/verify_openhands_wlasl_lstm.py`
  - checkpoint load verification helper
- `scripts/run_openhands_wlasl_lstm.py`
  - local model runner helper
- `external/OpenHands`
  - cloned upstream OpenHands repo used for model/runtime code

## Requirements

Frontend dependency:

- `@mediapipe/tasks-vision`

Backend runtime currently assumes availability of:

- `python3`
- `torch`
- `omegaconf`
- OpenHands import path from `external/OpenHands`

The repository also expects these local assets to exist:

- `models/hand_landmarker.task`
- `models/pose_landmarker_lite.task`
- OpenHands checkpoint and metadata under `assets/openhands/`

## Run locally

This app should be run through `server.py`, not `python3 -m http.server`, because the current frontend depends on the local inference endpoint.

From the project root:

```bash
python3 server.py
```

Then open:

```text
http://127.0.0.1:4173
```

If needed, you can also use:

```text
http://localhost:4173
```

Camera access requires running on `localhost`/`127.0.0.1` or over `https`.

## API

### `POST /api/infer`

Request body:

```json
{
  "frames": [
    [
      { "x": 0.1, "y": 0.2 },
      { "x": 0.2, "y": 0.3 }
    ]
  ]
}
```

Actual requirements:

- `frames` must be a non-empty array
- each frame must contain exactly 27 points
- each point is expected to contain numeric `x` and `y`
- at least 8 frames are required for inference

Response shape:

```json
{
  "topPrediction": {
    "gloss": "hello",
    "score": 0.42
  },
  "predictions": [
    { "gloss": "hello", "score": 0.42 }
  ],
  "framesUsed": 36
}
```

## Known issues

- The current model is not accurate enough for dependable real-world communication.
- The app labels output as “translated,” but most model predictions are not truly translated unless they match one of the demo phrases.
- The UI still mixes real inference with demo content.
- The landmark overlay currently emphasizes hands and does not fully visualize the 27-point inference tensor.
- Inference happens only after recording stops; there is no true streaming decode.
- The backend runs on CPU and may be slow on weaker machines.
- No automated tests are configured in `package.json`.

## Recommended usage right now

Use the current app as:

- a proof of local sign-capture plumbing
- a proof that a public OpenHands checkpoint can be called from the browser flow
- a prototype for isolated-word experimentation

Do not treat it as:

- a production ASL translator
- a complete accessibility communication tool
- a benchmark for real-world ASL quality

## Next practical improvements

The highest-value next changes are:

1. Remove or clearly separate demo translation behavior from real model output.
2. Show top-3 or top-5 predictions in the UI instead of only the top gloss.
3. Add a better translation backend for arbitrary English gloss text.
4. Add a second recognizer path for fingerspelling.
5. Replace the current isolated-word baseline with a stronger open-source checkpoint if a usable one is confirmed.

## Sources

- OpenHands: [https://github.com/AI4Bharat/OpenHands](https://github.com/AI4Bharat/OpenHands)
- OpenHands paper: [https://arxiv.org/abs/2110.05877](https://arxiv.org/abs/2110.05877)
- WLASL dataset: [https://dxli94.github.io/WLASL/](https://dxli94.github.io/WLASL/)
