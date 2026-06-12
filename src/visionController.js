import { AslSequenceModel } from "./modelAdapter.js";

const MEDIAPIPE_IMPORT_URL = "../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
const WASM_ROOT = "../node_modules/@mediapipe/tasks-vision/wasm";
const HAND_MODEL_URL = "../models/hand_landmarker.task";
const POSE_MODEL_URL = "../models/pose_landmarker_lite.task";

const HAND_PICK_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const POSE_PICK_INDICES = [0, 2, 5, 11, 12, 13, 14];

function toPoint(landmark) {
  if (!landmark) {
    return { x: 0, y: 0, z: 0 };
  }

  return {
    x: landmark.x ?? 0,
    y: landmark.y ?? 0,
    z: landmark.z ?? 0,
  };
}

function selectPoints(points, indices) {
  return indices.map((index) => toPoint(points?.[index]));
}

function handednessScore(handedness = []) {
  return handedness
    .flat()
    .reduce((maxScore, item) => Math.max(maxScore, item?.score ?? 0), 0);
}

function assignHands(landmarks = [], handedness = []) {
  const slots = {
    Left: null,
    Right: null,
  };

  for (let index = 0; index < landmarks.length; index += 1) {
    const handClass = handedness[index]?.[0]?.categoryName;
    if (handClass === "Left" || handClass === "Right") {
      slots[handClass] = landmarks[index];
    }
  }

  return slots;
}

function buildOpenHandsPoints27(poseLandmarks, handLandmarks, handedness) {
  const posePoints = selectPoints(poseLandmarks, POSE_PICK_INDICES);
  const hands = assignHands(handLandmarks, handedness);
  const leftHandPoints = selectPoints(hands.Left, HAND_PICK_INDICES);
  const rightHandPoints = selectPoints(hands.Right, HAND_PICK_INDICES);
  return [...posePoints, ...leftHandPoints, ...rightHandPoints];
}

export class VisionController {
  constructor() {
    this.mode = "booting";
    this.handLandmarker = null;
    this.poseLandmarker = null;
    this.sequenceModel = new AslSequenceModel();
    this.lastVideoTime = -1;
  }

  async init() {
    try {
      this.mode = "loading-mediapipe";
      const visionModule = await import(MEDIAPIPE_IMPORT_URL);
      const { FilesetResolver, HandLandmarker, PoseLandmarker } = visionModule;
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: POSE_MODEL_URL,
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

      this.mode = "openhands-pose-capture";
      return {
        mode: this.mode,
        hint: "OpenHands-compatible pose capture is active. Record a single ASL word, then stop to run the downloaded WLASL model.",
      };
    } catch (error) {
      this.mode = "fallback";
      return {
        mode: this.mode,
        hint: `MediaPipe failed to load. ${String(error)}`,
      };
    }
  }

  reset() {
    this.sequenceModel.reset();
    this.lastVideoTime = -1;
  }

  getSequence() {
    return this.sequenceModel.getSequence();
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
    const points27 = buildOpenHandsPoints27(
      poseLandmarks,
      handLandmarks,
      handedness,
    );
    const sequenceLength = this.sequenceModel.pushFrame(points27);

    return {
      handsDetected: handLandmarks.length,
      sequenceLength,
      confidence: handednessScore(handedness),
      points27,
      overlayHands: handLandmarks.map((hand) => hand.map(toPoint)),
      overlayPose: POSE_PICK_INDICES.map((index) => toPoint(poseLandmarks[index])),
      mode: this.mode,
    };
  }
}
