import { TranslationService } from "./translationService.js";
import { SUPPORTED_LANGUAGES } from "./demoPhrases.js";
import { VisionController } from "./visionController.js";

const PHRASE_IDLE_RESET_MS = 4000; // start a new phrase after 4 s of silence

const elements = {
  camera: document.querySelector("#camera"),
  overlay: document.querySelector("#overlay"),
  videoEmptyState: document.querySelector("#videoEmptyState"),
  startCameraBtn: document.querySelector("#startCameraBtn"),
  stopCameraBtn: document.querySelector("#stopCameraBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  speakBtn: document.querySelector("#speakBtn"),
  languageSelect: document.querySelector("#languageSelect"),
  englishTranscript: document.querySelector("#englishTranscript"),
  translatedTranscript: document.querySelector("#translatedTranscript"),
  sessionLog: document.querySelector("#sessionLog"),
  recognitionState: document.querySelector("#recognitionState"),
  gestureToken: document.querySelector("#gestureToken"),
  confidenceValue: document.querySelector("#confidenceValue"),
  cameraStatus: document.querySelector("#cameraStatus"),
  cameraDot: document.querySelector("#cameraDot"),
  engineMode: document.querySelector("#engineMode"),
  engineHint: document.querySelector("#engineHint"),
};

const state = {
  stream: null,
  frameLoopId: 0,
  selectedLanguage: "hi",
  pendingInference: false,
  phraseGlosses: [],    // accumulated glosses in current phrase
  lastWordTime: 0,
  latestEnglish: "",
};

const translationService = new TranslationService();
const drawingContext = elements.overlay.getContext("2d");

// ── inference callback ────────────────────────────────────────────────────────

async function handleSignReady(frames) {
  if (state.pendingInference) return; // drop overlapping calls
  state.pendingInference = true;
  elements.recognitionState.textContent = "Running inference…";

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    const payload = await response.json();

    if (!response.ok) throw new Error(payload.error ?? "Inference failed");

    const { gloss, score } = payload.topPrediction;
    elements.gestureToken.textContent = gloss;
    elements.confidenceValue.textContent = `${Math.round(score * 100)}%`;

    // Accumulate into current phrase, resetting after a long pause
    const now = Date.now();
    if (state.phraseGlosses.length > 0 && now - state.lastWordTime > PHRASE_IDLE_RESET_MS) {
      finalisePhrase();
    }
    state.phraseGlosses.push(gloss);
    state.lastWordTime = now;

    const phraseText = state.phraseGlosses.join(" ");
    state.latestEnglish = phraseText;
    elements.englishTranscript.textContent = phraseText;

    const translated = await translationService.translate(phraseText, state.selectedLanguage);
    elements.translatedTranscript.textContent = translated;
    elements.speakBtn.disabled = false;
    elements.recognitionState.textContent = "Watching…";
  } catch (error) {
    elements.recognitionState.textContent = "Inference error";
    elements.englishTranscript.textContent = String(error.message ?? error);
  } finally {
    state.pendingInference = false;
  }
}

function finalisePhrase() {
  if (!state.phraseGlosses.length) return;
  const text = state.phraseGlosses.join(" ");
  const translated = elements.translatedTranscript.textContent;
  addLogEntry(text, translated);
  state.phraseGlosses = [];
}

// ── gesture callback ──────────────────────────────────────────────────────────

function handleGestureDetected({ gesture, score }) {
  elements.gestureToken.textContent = gesture;
  elements.confidenceValue.textContent = `${Math.round(score * 100)}%`;
}

// ── vision controller ─────────────────────────────────────────────────────────

const visionController = new VisionController({
  onSignReady: handleSignReady,
  onGestureDetected: handleGestureDetected,
});

// ── UI helpers ────────────────────────────────────────────────────────────────

function populateLanguageSelect() {
  elements.languageSelect.innerHTML = SUPPORTED_LANGUAGES.map(
    ({ code, label }) => `<option value="${code}">${label}</option>`,
  ).join("");
  elements.languageSelect.value = state.selectedLanguage;
}

function resizeCanvasToVideo() {
  const { videoWidth: w, videoHeight: h } = elements.camera;
  if (w && h) {
    elements.overlay.width = w;
    elements.overlay.height = h;
  }
}

const STATE_COLORS = {
  idle: "rgba(255,255,255,0.6)",
  signing: "rgba(201,95,55,0.9)",
};

function drawOverlay(landmarkSets = [], detectorState = "idle") {
  drawingContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  drawingContext.save();
  drawingContext.scale(-1, 1);
  drawingContext.translate(-elements.overlay.width, 0);

  // Guide box — orange while signing, white otherwise
  drawingContext.strokeStyle = STATE_COLORS[detectorState] ?? STATE_COLORS.idle;
  drawingContext.lineWidth = detectorState === "signing" ? 4 : 2;
  drawingContext.strokeRect(
    elements.overlay.width * 0.18,
    elements.overlay.height * 0.12,
    elements.overlay.width * 0.64,
    elements.overlay.height * 0.76,
  );

  drawingContext.fillStyle = STATE_COLORS.signing;
  for (const hand of landmarkSets) {
    for (const point of hand) {
      drawingContext.beginPath();
      drawingContext.arc(
        point.x * elements.overlay.width,
        point.y * elements.overlay.height,
        4, 0, Math.PI * 2,
      );
      drawingContext.fill();
    }
  }
  drawingContext.restore();
}

function setCameraLive(isLive) {
  elements.cameraDot.classList.toggle("live", isLive);
  elements.cameraStatus.textContent = isLive ? "Camera live" : "Camera idle";
  elements.videoEmptyState.hidden = isLive;
  elements.startCameraBtn.disabled = isLive;
  elements.stopCameraBtn.disabled = !isLive;
}

function addLogEntry(english, translated) {
  const item = document.createElement("li");
  const ts = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  item.innerHTML = `<small>${ts}</small>${english}<br />${translated}`;
  elements.sessionLog.prepend(item);
}

// ── frame loop ────────────────────────────────────────────────────────────────

async function handleFrame() {
  if (!state.stream) return;

  if (elements.camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const result = visionController.capture(elements.camera, performance.now());

    if (result) {
      drawOverlay(result.overlayHands, result.detectorState);

      // Status line
      if (!state.pendingInference) {
        if (result.detectorState === "signing") {
          elements.recognitionState.textContent = `Signing — ${result.buffered} frames`;
        } else if (result.handsDetected > 0) {
          elements.recognitionState.textContent = "Hands detected — watching…";
        } else {
          elements.recognitionState.textContent = "Watching for signs…";
        }
      }
    }
  }

  state.frameLoopId = window.requestAnimationFrame(handleFrame);
}

// ── camera ────────────────────────────────────────────────────────────────────

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    state.stream = stream;
    elements.camera.srcObject = stream;
    await elements.camera.play();
    resizeCanvasToVideo();
    drawOverlay();
    setCameraLive(true);
    elements.recognitionState.textContent = "Watching for signs…";
    handleFrame();
  } catch (error) {
    elements.recognitionState.textContent = "Camera access failed";
    elements.englishTranscript.textContent = "Camera permission was denied or unavailable.";
    console.error(error);
  }
}

function stopCamera() {
  if (state.frameLoopId) {
    window.cancelAnimationFrame(state.frameLoopId);
    state.frameLoopId = 0;
  }
  visionController.reset();
  finalisePhrase();

  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
  }
  state.stream = null;
  elements.camera.srcObject = null;
  drawingContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  setCameraLive(false);
  elements.recognitionState.textContent = "Waiting for camera";
}

// ── clear / speak ─────────────────────────────────────────────────────────────

function clearSession() {
  visionController.reset();
  state.phraseGlosses = [];
  state.latestEnglish = "";
  elements.englishTranscript.textContent = "No sign detected yet.";
  elements.translatedTranscript.textContent = "Signs will appear here as you sign.";
  elements.gestureToken.textContent = "—";
  elements.confidenceValue.textContent = "—";
  elements.sessionLog.innerHTML = "";
  elements.speakBtn.disabled = true;
  drawOverlay();
}

function speakCurrentTranslation() {
  const text = elements.translatedTranscript.textContent;
  if (!text || text === "Signs will appear here as you sign.") return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = state.selectedLanguage;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

// ── event listeners ───────────────────────────────────────────────────────────

elements.startCameraBtn.addEventListener("click", startCamera);
elements.stopCameraBtn.addEventListener("click", stopCamera);
elements.clearBtn.addEventListener("click", clearSession);
elements.speakBtn.addEventListener("click", speakCurrentTranslation);
elements.languageSelect.addEventListener("change", async (event) => {
  state.selectedLanguage = event.target.value;
  if (state.latestEnglish) {
    const translated = await translationService.translate(
      state.latestEnglish,
      state.selectedLanguage,
    );
    elements.translatedTranscript.textContent = translated;
  }
});

window.addEventListener("resize", () => {
  resizeCanvasToVideo();
  drawOverlay();
});
window.addEventListener("beforeunload", stopCamera);

// ── boot ──────────────────────────────────────────────────────────────────────

populateLanguageSelect();
elements.engineMode.textContent = "loading";
elements.engineHint.textContent = "Initialising MediaPipe…";

visionController.init().then((status) => {
  elements.engineMode.textContent = status.mode;
  elements.engineHint.textContent = status.hint;
  elements.recognitionState.textContent =
    status.mode === "continuous" ? "Waiting for camera" : "Recognition unavailable";
  clearSession();
});
