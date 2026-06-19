import { TranslationService } from "./translationService.js";
import { SUPPORTED_LANGUAGES } from "./demoPhrases.js";
import { VisionController } from "./visionController.js";

const PHRASE_IDLE_RESET_MS = 4000;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const elements = {
  camera: document.querySelector("#camera"),
  overlay: document.querySelector("#overlay"),
  videoEmptyState: document.querySelector("#videoEmptyState"),
  startCameraBtn: document.querySelector("#startCameraBtn"),
  stopCameraBtn: document.querySelector("#stopCameraBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  speakBtn: document.querySelector("#speakBtn"),
  languageSelect: document.querySelector("#languageSelect"),
  modeSelect: document.querySelector("#modeSelect"),
  englishTranscript: document.querySelector("#englishTranscript"),
  translatedTranscript: document.querySelector("#translatedTranscript"),
  sessionLog: document.querySelector("#sessionLog"),
  recognitionState: document.querySelector("#recognitionState"),
  gestureToken: document.querySelector("#gestureToken"),
  confidenceValue: document.querySelector("#confidenceValue"),
  handsValue: document.querySelector("#handsValue"),
  velocityValue: document.querySelector("#velocityValue"),
  cameraStatus: document.querySelector("#cameraStatus"),
  cameraDot: document.querySelector("#cameraDot"),
  engineMode: document.querySelector("#engineMode"),
  engineHint: document.querySelector("#engineHint"),
};

const state = {
  stream: null,
  frameLoopId: 0,
  selectedLanguage: "hi",
  mode: "signs",           // "signs" | "fingerspell"
  pendingInference: false,
  phraseTokens: [],
  lastWordTime: 0,
  latestEnglish: "",
};

const translationService = new TranslationService();
const drawingContext = elements.overlay.getContext("2d");

// ── inference ─────────────────────────────────────────────────────────────────

async function handleSignReady(frames27, fullFrames) {
  if (state.pendingInference) return;
  state.pendingInference = true;

  const isFS = state.mode === "fingerspell";
  elements.recognitionState.textContent = isFS ? "Decoding fingerspelling…" : "Running inference…";

  try {
    let word = null;
    let confidence = null;

    if (isFS) {
      const resp = await fetch("/api/fingerspell", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames: fullFrames }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error ?? "Fingerspelling failed");
      word = payload.text;
      confidence = null; // Squeezeformer doesn't emit a score
    } else {
      const resp = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames: frames27 }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error ?? "Inference failed");
      word = payload.topPrediction.gloss;
      confidence = payload.topPrediction.score;
    }

    if (!word) {
      elements.recognitionState.textContent = "Watching…";
      return;
    }

    elements.gestureToken.textContent = word;
    elements.confidenceValue.textContent = confidence !== null
      ? `${Math.round(confidence * 100)}%`
      : "—";

    // Phrase accumulation: reset after idle gap
    const now = Date.now();
    if (state.phraseTokens.length > 0 && now - state.lastWordTime > PHRASE_IDLE_RESET_MS) {
      finalisePhrase();
    }
    state.phraseTokens.push(word);
    state.lastWordTime = now;

    const phraseText = state.phraseTokens.join(" ");
    state.latestEnglish = phraseText;
    elements.englishTranscript.textContent = phraseText;

    const translated = await translationService.translate(phraseText, state.selectedLanguage);
    elements.translatedTranscript.textContent = translated;
    elements.speakBtn.disabled = false;
    elements.recognitionState.textContent = "Watching…";
  } catch (err) {
    elements.recognitionState.textContent = "Inference error";
    elements.englishTranscript.textContent = String(err.message ?? err);
  } finally {
    state.pendingInference = false;
  }
}

function finalisePhrase() {
  if (!state.phraseTokens.length) return;
  addLogEntry(state.phraseTokens.join(" "), elements.translatedTranscript.textContent);
  state.phraseTokens = [];
}

// ── gesture callback ──────────────────────────────────────────────────────────

function handleGestureDetected({ gesture, score }) {
  // Only surface this in signs mode; in fingerspell mode hand shapes are letter candidates
  if (state.mode !== "signs") return;
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
  if (w && h) { elements.overlay.width = w; elements.overlay.height = h; }
}

function drawOverlay(hands = [], detectorState = "idle", velocity = 0, handsCount = 0) {
  const W = elements.overlay.width;
  const H = elements.overlay.height;
  drawingContext.clearRect(0, 0, W, H);
  drawingContext.save();
  drawingContext.scale(-1, 1);
  drawingContext.translate(-W, 0);

  const signing = detectorState === "signing";
  const hasHands = handsCount > 0;

  // Bounding box — orange when signing, green when hands visible, white when idle
  drawingContext.strokeStyle = signing
    ? "rgba(201,95,55,0.95)"
    : hasHands
    ? "rgba(45,175,90,0.8)"
    : "rgba(255,255,255,0.35)";
  drawingContext.lineWidth = signing ? 4 : hasHands ? 3 : 1.5;
  drawingContext.strokeRect(W * 0.18, H * 0.12, W * 0.64, H * 0.76);

  // Hand skeleton
  const skelColor = signing ? "rgba(201,95,55,0.85)" : "rgba(45,220,100,0.85)";
  const dotColor  = signing ? "rgba(255,140,80,1)"    : "rgba(80,255,140,1)";

  for (const hand of hands) {
    // Connections
    drawingContext.strokeStyle = skelColor;
    drawingContext.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = hand[a]; const pb = hand[b];
      if (!pa || !pb) continue;
      drawingContext.beginPath();
      drawingContext.moveTo(pa.x * W, pa.y * H);
      drawingContext.lineTo(pb.x * W, pb.y * H);
      drawingContext.stroke();
    }
    // Landmark dots
    drawingContext.fillStyle = dotColor;
    for (const pt of hand) {
      drawingContext.beginPath();
      drawingContext.arc(pt.x * W, pt.y * H, 4, 0, Math.PI * 2);
      drawingContext.fill();
    }
  }

  // Velocity bar along bottom of video
  if (hasHands) {
    const barW = W * 0.5;
    const barX = W * 0.25;
    const barY = H - 22;
    const fillRatio = Math.min(velocity / 0.05, 1);
    // Background track
    drawingContext.fillStyle = "rgba(0,0,0,0.35)";
    roundRect(drawingContext, barX, barY, barW, 8, 4);
    drawingContext.fill();
    // Fill
    if (fillRatio > 0) {
      drawingContext.fillStyle = signing ? "rgba(201,95,55,0.95)" : "rgba(45,220,100,0.95)";
      roundRect(drawingContext, barX, barY, barW * fillRatio, 8, 4);
      drawingContext.fill();
    }
    // Threshold tick
    const tickX = barX + (0.007 / 0.05) * barW;
    drawingContext.strokeStyle = "rgba(255,255,255,0.9)";
    drawingContext.lineWidth = 1.5;
    drawingContext.beginPath();
    drawingContext.moveTo(tickX, barY - 3);
    drawingContext.lineTo(tickX, barY + 11);
    drawingContext.stroke();
  }

  drawingContext.restore();

  // "HANDS DETECTED" badge — drawn unmirrored
  if (hasHands) {
    drawingContext.save();
    const label = `${handsCount} hand${handsCount > 1 ? "s" : ""} detected`;
    drawingContext.font = "bold 12px 'IBM Plex Sans', sans-serif";
    const tw = drawingContext.measureText(label).width;
    const bx = W / 2 - tw / 2 - 8;
    const by = 18;
    drawingContext.fillStyle = signing ? "rgba(180,70,30,0.85)" : "rgba(30,120,60,0.85)";
    roundRect(drawingContext, bx, by, tw + 16, 22, 6);
    drawingContext.fill();
    drawingContext.fillStyle = "#fff";
    drawingContext.fillText(label, bx + 8, by + 15);
    drawingContext.restore();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
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
  const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const badge = `<span class="mode-badge">${state.mode === "fingerspell" ? "FS" : "ASL"}</span>`;
  item.innerHTML = `<small>${ts} ${badge}</small>${english}<br />${translated}`;
  elements.sessionLog.prepend(item);
}

// ── frame loop ────────────────────────────────────────────────────────────────

async function handleFrame() {
  if (!state.stream) return;

  if (elements.camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const result = visionController.capture(elements.camera, performance.now());

    if (result) {
      drawOverlay(result.overlayHands, result.detectorState, result.velocity, result.handsDetected);

      if (!state.pendingInference) {
        if (result.detectorState === "signing") {
          const label = state.mode === "fingerspell" ? "Fingerspelling" : "Signing";
          elements.recognitionState.textContent = `${label} — ${result.buffered} frames buffered`;
        } else if (result.handsDetected > 0) {
          elements.recognitionState.textContent = "Hands visible — move to sign, hold to complete";
        } else {
          elements.recognitionState.textContent = "No hands — show your hand(s) to the camera";
        }
        // live helpers
        elements.handsValue.textContent = result.handsDetected > 0
          ? `${result.handsDetected} detected`
          : "none";
        elements.velocityValue.textContent = (result.velocity * 1000).toFixed(1);
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
  } catch (err) {
    elements.recognitionState.textContent = "Camera access failed";
    elements.englishTranscript.textContent = "Camera permission denied or unavailable.";
  }
}

function stopCamera() {
  if (state.frameLoopId) { window.cancelAnimationFrame(state.frameLoopId); state.frameLoopId = 0; }
  visionController.reset();
  finalisePhrase();
  if (state.stream) { for (const t of state.stream.getTracks()) t.stop(); }
  state.stream = null;
  elements.camera.srcObject = null;
  drawingContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  setCameraLive(false);
  elements.recognitionState.textContent = "Waiting for camera";
}

function clearSession() {
  visionController.reset();
  state.phraseTokens = [];
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
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = state.selectedLanguage;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utt);
}

// ── event listeners ───────────────────────────────────────────────────────────

elements.startCameraBtn.addEventListener("click", startCamera);
elements.stopCameraBtn.addEventListener("click", stopCamera);
elements.clearBtn.addEventListener("click", clearSession);
elements.speakBtn.addEventListener("click", speakCurrentTranslation);

elements.modeSelect.addEventListener("change", (e) => {
  state.mode = e.target.value;
  visionController.reset();
  const label = state.mode === "fingerspell" ? "Fingerspelling mode" : "Word-sign mode";
  elements.recognitionState.textContent = label + " — watching…";
});

elements.languageSelect.addEventListener("change", async (e) => {
  state.selectedLanguage = e.target.value;
  if (state.latestEnglish) {
    elements.translatedTranscript.textContent = await translationService.translate(
      state.latestEnglish, state.selectedLanguage,
    );
  }
});

window.addEventListener("resize", () => { resizeCanvasToVideo(); drawOverlay(); });
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
