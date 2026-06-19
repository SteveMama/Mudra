import { SignBoundaryDetector } from "./modelAdapter.js";

const MEDIAPIPE_IMPORT_URL = "../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
const WASM_ROOT = "../node_modules/@mediapipe/tasks-vision/wasm";
const HAND_MODEL_URL = "../models/hand_landmarker.task";
const POSE_MODEL_URL = "../models/pose_landmarker_lite.task";
const GESTURE_MODEL_URL = "../models/gesture_recognizer.task";

const HAND_PICK_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const POSE_PICK_INDICES = [0, 2, 5, 11, 12, 13, 14];

// Gestures the built-in model doesn't suppress
const INFORMATIVE_GESTURES = new Set([
  "Closed_Fist",
  "Open_Palm",
  "Pointing_Up",
  "Thumb_Down",
  "Thumb_Up",
  "Victory",
  "ILoveYou",
]);

function toPoint(landmark) {
  if (!landmark) return { x: 0, y: 0, z: 0 };
  return { x: landmark.x ?? 0, y: landmark.y ?? 0, z: landmark.z ?? 0 };
}

function selectPoints(points, indices) {
  return indices.map((i) => toPoint(points?.[i]));
}

function handednessScore(handedness = []) {
  return handedness
    .flat()
    .reduce((max, item) => Math.max(max, item?.score ?? 0), 0);
}

function assignHands(landmarks = [], handedness = []) {
  const slots = { Left: null, Right: null };
  for (let i = 0; i < landmarks.length; i++) {
    const cls = handedness[i]?.[0]?.categoryName;
    if (cls === "Left" || cls === "Right") slots[cls] = landmarks[i];
  }
  return slots;
}

function buildOpenHandsPoints27(poseLandmarks, handLandmarks, handedness) {
  const posePoints = selectPoints(poseLandmarks, POSE_PICK_INDICES);
  const hands = assignHands(handLandmarks, handedness);
  const leftPoints = selectPoints(hands.Left, HAND_PICK_INDICES);
  const rightPoints = selectPoints(hands.Right, HAND_PICK_INDICES);
  return [...posePoints, ...leftPoints, ...rightPoints];
}

export class VisionController {
  constructor({ onSignReady, onGestureDetected } = {}) {
    this.mode = "booting";
    this.handLandmarker = null;
    this.poseLandmarker = null;
    this.gestureRecognizer = null;
    this.boundaryDetector = new SignBoundaryDetector(onSignReady ?? (() => {}));
    this.onGestureDetected = onGestureDetected ?? (() => {});
    this.lastVideoTime = -1;
  }

  async init() {
    try {
      this.mode = "loading-mediapipe";
      const visionModule = await import(MEDIAPIPE_IMPORT_URL);
      const { FilesetResolver, HandLandmarker, PoseLandmarker, GestureRecognizer } =
        visionModule;
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      // Gesture recognizer is optional — load best-effort for gesture/letter detection
      try {
        this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: { modelAssetPath: GESTURE_MODEL_URL },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch {
        console.warn("[mudra] GestureRecognizer failed to load — gesture layer disabled");
      }

      this.mode = "continuous";
      return {
        mode: this.mode,
        hint: "Continuous sign detection active. Sign in view of the camera.",
      };
    } catch (error) {
      this.mode = "fallback";
      return {
        mode: this.mode,
        hint: `MediaPipe failed to load: ${String(error)}`,
      };
    }
  }

  reset() {
    this.boundaryDetector.reset();
    this.lastVideoTime = -1;
  }

  capture(videoElement, timestampMs) {
    if (
      !this.handLandmarker ||
      !this.poseLandmarker ||
      videoElement.currentTime === this.lastVideoTime
    ) {
      return null;
    }
    this.lastVideoTime = videoElement.currentTime;

    const handResult = this.handLandmarker.detectForVideo(videoElement, timestampMs);
    const poseResult = this.poseLandmarker.detectForVideo(videoElement, timestampMs);

    const handLandmarks = handResult?.landmarks ?? [];
    const poseLandmarks = poseResult?.landmarks?.[0] ?? [];
    const handedness = handResult?.handedness ?? [];

    // Gesture/letter detection (best-effort)
    if (this.gestureRecognizer && handLandmarks.length > 0) {
      const gr = this.gestureRecognizer.recognizeForVideo(videoElement, timestampMs);
      const topGesture = (gr?.gestures ?? [])
        .flat()
        .filter((g) => INFORMATIVE_GESTURES.has(g.categoryName))
        .sort((a, b) => b.score - a.score)[0];
      if (topGesture && topGesture.score > 0.85) {
        this.onGestureDetected({ gesture: topGesture.categoryName, score: topGesture.score });
      }
    }

    const points27 = buildOpenHandsPoints27(poseLandmarks, handLandmarks, handedness);
    const boundary = this.boundaryDetector.pushFrame(points27, handLandmarks.length);

    return {
      handsDetected: handLandmarks.length,
      detectorState: boundary.state,
      buffered: boundary.buffered,
      velocity: boundary.velocity ?? 0,
      confidence: handednessScore(handedness),
      overlayHands: handLandmarks.map((hand) => hand.map(toPoint)),
      overlayPose: POSE_PICK_INDICES.map((i) => toPoint(poseLandmarks[i])),
      mode: this.mode,
    };
  }
}
