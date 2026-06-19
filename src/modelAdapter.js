// Wrist + fingertip indices within the 27-point skeleton.
// Left wrist=7, left fingertips=10,12,14,16; right wrist=17, right fingertips=20,22,24,26
const MOTION_POINTS = [7, 10, 12, 14, 16, 17, 20, 22, 24, 26];

const HOLD_VELOCITY_THRESHOLD = 0.007;
const HOLD_FRAMES_REQUIRED = 12;
const MIN_SIGN_FRAMES = 8;
const MAX_SIGN_FRAMES = 150;

function frameVelocity(prev, curr) {
  let sum = 0;
  for (const i of MOTION_POINTS) {
    const dx = curr[i].x - prev[i].x;
    const dy = curr[i].y - prev[i].y;
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / MOTION_POINTS.length;
}

export class SignBoundaryDetector {
  constructor(onSignReady) {
    this.onSignReady = onSignReady;
    this._state = "idle";
    this._buffer = [];
    this._holdCount = 0;
    this._prevFrame = null;
  }

  get state() {
    return this._state;
  }

  reset() {
    this._state = "idle";
    this._buffer = [];
    this._holdCount = 0;
    this._prevFrame = null;
  }

  // Returns { state, buffered, velocity }
  pushFrame(points27, handsDetected) {
    if (!this._prevFrame) {
      this._prevFrame = points27;
      return { state: this._state, buffered: 0, velocity: 0 };
    }

    const velocity = frameVelocity(this._prevFrame, points27);
    this._prevFrame = points27;
    const moving = velocity > HOLD_VELOCITY_THRESHOLD && handsDetected > 0;

    if (this._state === "idle") {
      if (moving) {
        this._state = "signing";
        this._buffer = [points27];
        this._holdCount = 0;
      }
    } else {
      // "signing" state — collect frames
      this._buffer.push(points27);

      if (!moving) {
        this._holdCount++;
        if (this._holdCount >= HOLD_FRAMES_REQUIRED) {
          this._fire();
        }
      } else {
        this._holdCount = 0;
      }

      if (this._buffer.length >= MAX_SIGN_FRAMES) {
        this._fire();
      }
    }

    return { state: this._state, buffered: this._buffer.length, velocity };
  }

  _fire() {
    const frames = this._buffer.slice();
    this._state = "idle";
    this._buffer = [];
    this._holdCount = 0;
    if (frames.length >= MIN_SIGN_FRAMES) {
      this.onSignReady(frames);
    }
  }
}
