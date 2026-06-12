import { TranslationService } from "./translationService.js";
import { DEMO_PHRASES, SUPPORTED_LANGUAGES } from "./demoPhrases.js";
import { VisionController } from "./visionController.js";

const elements = {
  camera: document.querySelector("#camera"),
  overlay: document.querySelector("#overlay"),
  videoEmptyState: document.querySelector("#videoEmptyState"),
  startCameraBtn: document.querySelector("#startCameraBtn"),
  stopCameraBtn: document.querySelector("#stopCameraBtn"),
  recordBtn: document.querySelector("#recordBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  speakBtn: document.querySelector("#speakBtn"),
  languageSelect: document.querySelector("#languageSelect"),
  demoPhraseSelect: document.querySelector("#demoPhraseSelect"),
  runDemoBtn: document.querySelector("#runDemoBtn"),
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
  recording: false,
  frameLoopId: 0,
  selectedLanguage: "hi",
  latestEnglish: "",
  visionReady: false,
  pendingInference: false,
};

const translationService = new TranslationService();
const visionController = new VisionController();
const drawingContext = elements.overlay.getContext("2d");

function populateLanguageSelect() {
  const options = SUPPORTED_LANGUAGES.map(
    ({ code, label }) => `<option value="${code}">${label}</option>`,
  );
  elements.languageSelect.innerHTML = options.join("");
  elements.languageSelect.value = state.selectedLanguage;
}

function populateDemoPhraseSelect() {
  const options = DEMO_PHRASES.map(
    (phrase) => `<option value="${phrase.id}">${phrase.english}</option>`,
  );
  elements.demoPhraseSelect.innerHTML = options.join("");
}

function resizeCanvasToVideo() {
  const width = elements.camera.videoWidth;
  const height = elements.camera.videoHeight;

  if (!width || !height) {
    return;
  }

  elements.overlay.width = width;
  elements.overlay.height = height;
}

function drawStatusOverlay(landmarkSets = []) {
  drawingContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  drawingContext.save();
  drawingContext.scale(-1, 1);
  drawingContext.translate(-elements.overlay.width, 0);
  drawingContext.strokeStyle = "rgba(255, 248, 243, 0.85)";
  drawingContext.lineWidth = 3;
  drawingContext.strokeRect(
    elements.overlay.width * 0.18,
    elements.overlay.height * 0.12,
    elements.overlay.width * 0.64,
    elements.overlay.height * 0.76,
  );

  drawingContext.fillStyle = "rgba(201, 95, 55, 0.9)";
  for (const hand of landmarkSets) {
    for (const point of hand) {
      const x = point.x * elements.overlay.width;
      const y = point.y * elements.overlay.height;
      drawingContext.beginPath();
      drawingContext.arc(x, y, 4, 0, Math.PI * 2);
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
  elements.recordBtn.disabled = !isLive;
}

async function updateTranslation(english) {
  state.latestEnglish = english;
  const translated = await translationService.translate(
    english,
    state.selectedLanguage,
  );
  elements.englishTranscript.textContent = english;
  elements.translatedTranscript.textContent = translated;
  elements.speakBtn.disabled = false;
}

function addLogEntry(english, translated, confidence) {
  const item = document.createElement("li");
  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  item.innerHTML = `<small>${timestamp} • ${Math.round(
    confidence * 100,
  )}% confidence</small>${english}<br />${translated}`;
  elements.sessionLog.prepend(item);
}

async function commitPhrase(phrase, confidence = 1) {
  await updateTranslation(phrase.english);
  addLogEntry(
    phrase.english,
    elements.translatedTranscript.textContent,
    confidence,
  );
}

function processRecognitionResult(result) {
  if (!result) {
    return;
  }

  commitPhrase(result.phrase, result.confidence);
}

function showClassifierPendingState(sequenceLength) {
  elements.englishTranscript.textContent =
    "Capturing an OpenHands-compatible pose sequence.";
  elements.translatedTranscript.textContent =
    `Buffered ${sequenceLength} frame${sequenceLength === 1 ? "" : "s"} for the downloaded WLASL model.`;
  elements.speakBtn.disabled = true;
}

async function inferRecordedSequence() {
  const frames = visionController.getSequence();
  if (frames.length < 8) {
    elements.englishTranscript.textContent =
      "Recording was too short. Record a single clear ASL word for at least a second.";
    elements.translatedTranscript.textContent =
      "The OpenHands model needs a longer sequence.";
    return;
  }

  state.pendingInference = true;
  elements.recognitionState.textContent = "Running OpenHands inference";

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ frames }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Inference failed.");
    }

    elements.recognitionState.textContent = "Prediction ready";
    elements.gestureToken.textContent = String(frames.length);
    elements.confidenceValue.textContent = `${Math.round(
      payload.topPrediction.score * 100,
    )}%`;
    await updateTranslation(payload.topPrediction.gloss);
    addLogEntry(
      payload.topPrediction.gloss,
      elements.translatedTranscript.textContent,
      payload.topPrediction.score,
    );
  } catch (error) {
    elements.recognitionState.textContent = "Inference failed";
    elements.englishTranscript.textContent = String(error.message || error);
    elements.translatedTranscript.textContent =
      "The local OpenHands model could not return a prediction.";
  } finally {
    state.pendingInference = false;
  }
}

async function handleFrame() {
  if (!state.recording || !state.stream) {
    return;
  }

  if (elements.camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const recognition = visionController.capture(
      elements.camera,
      performance.now(),
    );

    if (recognition) {
      elements.gestureToken.textContent = String(recognition.handsDetected);
      elements.confidenceValue.textContent = String(recognition.sequenceLength);
      drawStatusOverlay(recognition.overlayHands);

      if (recognition.handsDetected > 0) {
        showClassifierPendingState(recognition.sequenceLength);
      }
    }
  }

  state.frameLoopId = window.requestAnimationFrame(handleFrame);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    state.stream = stream;
    elements.camera.srcObject = stream;
    await elements.camera.play();
    resizeCanvasToVideo();
    drawStatusOverlay();
    setCameraLive(true);
    elements.recognitionState.textContent = "Camera ready for OpenHands capture";
  } catch (error) {
    elements.recognitionState.textContent = "Camera access failed";
    elements.englishTranscript.textContent =
      "Camera permission was denied or unavailable.";
    console.error(error);
  }
}

function stopRecordingLoop() {
  if (state.frameLoopId) {
    window.cancelAnimationFrame(state.frameLoopId);
    state.frameLoopId = 0;
  }
}

function stopCamera() {
  stopRecordingLoop();
  state.recording = false;
  visionController.reset();

  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
  }

  state.stream = null;
  elements.camera.srcObject = null;
  drawingContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  setCameraLive(false);
  elements.recognitionState.textContent = "Waiting for camera";
  elements.recordBtn.textContent = "Record sign";
}

async function setRecording(recording) {
  state.recording = recording;
  elements.recordBtn.textContent = recording ? "Stop recording" : "Record sign";
  elements.recognitionState.textContent = recording
    ? "Capturing a single ASL word"
    : "Camera ready for OpenHands capture";

  if (recording) {
    clearSession();
    handleFrame();
  } else {
    stopRecordingLoop();
    await inferRecordedSequence();
  }
}

function clearSession() {
  visionController.reset();
  elements.englishTranscript.textContent = "No sign detected yet.";
  elements.translatedTranscript.textContent =
    "Record a single ASL word to query the downloaded OpenHands model.";
  elements.gestureToken.textContent = "0";
  elements.confidenceValue.textContent = "0";
  elements.sessionLog.innerHTML = "";
  elements.speakBtn.disabled = true;
  drawStatusOverlay();
}

function speakCurrentTranslation() {
  const text = elements.translatedTranscript.textContent;
  if (!text || text === "Translation will appear here.") {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = state.selectedLanguage;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function runSelectedDemoPhrase() {
  const phrase = DEMO_PHRASES.find(
    (entry) => entry.id === elements.demoPhraseSelect.value,
  );
  if (!phrase) {
    return;
  }

  elements.recognitionState.textContent = "Demo phrase committed";
  elements.gestureToken.textContent = "demo";
  elements.confidenceValue.textContent = "manual";
  commitPhrase(phrase, 1);
}

elements.startCameraBtn.addEventListener("click", startCamera);
elements.stopCameraBtn.addEventListener("click", stopCamera);
elements.recordBtn.addEventListener("click", async () => {
  if (!state.pendingInference) {
    await setRecording(!state.recording);
  }
});
elements.clearBtn.addEventListener("click", clearSession);
elements.speakBtn.addEventListener("click", speakCurrentTranslation);
elements.runDemoBtn.addEventListener("click", runSelectedDemoPhrase);
elements.languageSelect.addEventListener("change", async (event) => {
  state.selectedLanguage = event.target.value;
  if (state.latestEnglish) {
    await updateTranslation(state.latestEnglish);
  }
});

window.addEventListener("resize", () => {
  resizeCanvasToVideo();
  drawStatusOverlay();
});

window.addEventListener("beforeunload", () => {
  stopCamera();
});

populateLanguageSelect();
populateDemoPhraseSelect();
elements.engineMode.textContent = "loading-mediapipe";
elements.engineHint.textContent = "Loading OpenHands-compatible pose capture assets.";
visionController.init().then((status) => {
  state.visionReady = status.mode === "openhands-pose-capture";
  elements.engineMode.textContent = status.mode;
  elements.engineHint.textContent = status.hint;
  elements.recognitionState.textContent =
    status.mode === "openhands-pose-capture"
      ? "Waiting for camera"
      : "Recognition fallback active";
  clearSession();
});
