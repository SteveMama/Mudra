import { SignBoundaryDetector } from "./modelAdapter.js";

const MEDIAPIPE_IMPORT_URL = "../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
const WASM_ROOT = "../node_modules/@mediapipe/tasks-vision/wasm";
const HAND_MODEL_URL = "../models/hand_landmarker.task";
const POSE_MODEL_URL = "../models/pose_landmarker_lite.task";
const GESTURE_MODEL_URL = "../models/gesture_recognizer.task";
const FACE_MODEL_URL = "../models/face_landmarker.task";

// 27-point skeleton selection (matches OpenHands mediapipe_holistic_minimal_27 preset)
const HAND_PICK_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const POSE_PICK_INDICES = [0, 2, 5, 11, 12, 13, 14];

// Face landmark indices needed by the Kaggle fingerspelling model (76 specific points)
const FACE_INDICES_76 = [
  0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
  314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
  178, 87, 14, 317, 402, 318, 324, 308, 1, 2, 98, 327, 33, 7, 163, 144,
  145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390,
  373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
];

// Pose landmark indices needed by the Kaggle model (12 points: shoulder→ankle)
const POSE_INDICES_12 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];

// Gestures the built-in recognizer model understands (not ASL letters, but useful)
const INFORMATIVE_GESTURES = new Set([
  "Closed_Fist", "Open_Palm", "Pointing_Up",
  "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou",
]);

function toPoint(lm) {
  if (!lm) return { x: 0, y: 0, z: 0 };
  return { x: lm.x ?? 0, y: lm.y ?? 0, z: lm.z ?? 0 };
}

function selectPoints(pts, indices) {
  return indices.map((i) => toPoint(pts?.[i]));
}

function handednessScore(handedness = []) {
  return handedness.flat().reduce((m, it) => Math.max(m, it?.score ?? 0), 0);
}

function assignHands(landmarks = [], handedness = []) {
  const slots = { Left: null, Right: null };
  for (let i = 0; i < landmarks.length; i++) {
    const cls = handedness[i]?.[0]?.categoryName;
    if (cls === "Left" || cls === "Right") slots[cls] = landmarks[i];
  }
  return slots;
}

function buildPoints27(poseLandmarks, handLandmarks, handedness) {
  const hands = assignHands(handLandmarks, handedness);
  return [
    ...selectPoints(poseLandmarks, POSE_PICK_INDICES),
    ...selectPoints(hands.Left, HAND_PICK_INDICES),
    ...selectPoints(hands.Right, HAND_PICK_INDICES),
  ];
}

function buildFullFrame(poseLandmarks, handLandmarks, handedness, faceLandmarks) {
  const hands = assignHands(handLandmarks, handedness);
  return {
    // 76 face landmarks in FACE_INDICES_76 order (null if no FaceLandmarker)
    face76: faceLandmarks
      ? FACE_INDICES_76.map((i) => toPoint(faceLandmarks[i]))
      : null,
    // All 21 hand landmarks per hand
    leftHand: hands.Left ? hands.Left.map(toPoint) : Array(21).fill({ x: 0, y: 0, z: 0 }),
    rightHand: hands.Right ? hands.Right.map(toPoint) : Array(21).fill({ x: 0, y: 0, z: 0 }),
    // 12 pose landmarks (indices 11-22)
    pose12: POSE_INDICES_12.map((i) => toPoint(poseLandmarks?.[i])),
  };
}

export class VisionController {
  constructor({ onSignReady, onGestureDetected } = {}) {
    this.mode = "booting";
    this.handLandmarker = null;
    this.poseLandmarker = null;
    this.faceLandmarker = null;
    this.gestureRecognizer = null;
    this.hasFace = false;
    this.boundaryDetector = new SignBoundaryDetector(onSignReady ?? (() => {}));
    this.onGestureDetected = onGestureDetected ?? (() => {});
    this.lastVideoTime = -1;
  }

  async init() {
    try {
      this.mode = "loading-mediapipe";
      const mod = await import(MEDIAPIPE_IMPORT_URL);
      const { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker, GestureRecognizer } = mod;
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
      const cpu = (path) => ({ modelAssetPath: path, delegate: "CPU" });

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: cpu(HAND_MODEL_URL),
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: cpu(POSE_MODEL_URL),
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      // FaceLandmarker — needed for the fingerspelling model's face channel
      try {
        this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: cpu(FACE_MODEL_URL),
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
        this.hasFace = true;
      } catch {
        console.warn("[mudra] FaceLandmarker not available — fingerspelling face channel disabled");
      }

      // GestureRecognizer — optional, best-effort
      try {
        this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: cpu(GESTURE_MODEL_URL),
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch {
        console.warn("[mudra] GestureRecognizer not available");
      }

      this.mode = "continuous";
      return {
        mode: this.mode,
        hasFace: this.hasFace,
        hint: `Continuous detection active${this.hasFace ? " (face tracking on)" : " (no face model — fingerspelling face channel zeroed)"}.`,
      };
    } catch (error) {
      this.mode = "fallback";
      return { mode: this.mode, hasFace: false, hint: `MediaPipe failed: ${error}` };
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
    ) return null;
    this.lastVideoTime = videoElement.currentTime;

    const handResult = this.handLandmarker.detectForVideo(videoElement, timestampMs);
    const poseResult = this.poseLandmarker.detectForVideo(videoElement, timestampMs);

    const handLandmarks = handResult?.landmarks ?? [];
    const poseLandmarks = poseResult?.landmarks?.[0] ?? [];
    const handedness = handResult?.handedness ?? [];

    // Face landmarks (for fingerspelling model's face channel)
    let faceLandmarks = null;
    if (this.faceLandmarker && handLandmarks.length > 0) {
      const fr = this.faceLandmarker.detectForVideo(videoElement, timestampMs);
      faceLandmarks = fr?.faceLandmarks?.[0] ?? null;
    }

    // Gesture detection (best-effort, 8 built-in gestures)
    if (this.gestureRecognizer && handLandmarks.length > 0) {
      const gr = this.gestureRecognizer.recognizeForVideo(videoElement, timestampMs);
      const top = (gr?.gestures ?? [])
        .flat()
        .filter((g) => INFORMATIVE_GESTURES.has(g.categoryName))
        .sort((a, b) => b.score - a.score)[0];
      if (top && top.score > 0.85) {
        this.onGestureDetected({ gesture: top.categoryName, score: top.score });
      }
    }

    const points27 = buildPoints27(poseLandmarks, handLandmarks, handedness);
    const fullFrame = buildFullFrame(poseLandmarks, handLandmarks, handedness, faceLandmarks);

    const boundary = this.boundaryDetector.pushFrame(points27, handLandmarks.length, fullFrame);

    return {
      handsDetected: handLandmarks.length,
      hasFace: this.hasFace && faceLandmarks !== null,
      detectorState: boundary.state,
      buffered: boundary.buffered,
      velocity: boundary.velocity ?? 0,
      confidence: handednessScore(handedness),
      overlayHands: handLandmarks.map((h) => h.map(toPoint)),
      overlayPose: POSE_PICK_INDICES.map((i) => toPoint(poseLandmarks[i])),
      mode: this.mode,
    };
  }
}
