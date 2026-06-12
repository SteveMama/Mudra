const maxSequenceFrames = 180;

export class AslSequenceModel {
  constructor() {
    this.frames = [];
  }

  pushFrame(points27) {
    if (!Array.isArray(points27) || points27.length !== 27) {
      return this.frames.length;
    }

    this.frames.push(
      points27.map((point) => ({
        x: Number.isFinite(point?.x) ? point.x : 0,
        y: Number.isFinite(point?.y) ? point.y : 0,
      })),
    );
    this.frames = this.frames.slice(-maxSequenceFrames);
    return this.frames.length;
  }

  getSequence() {
    return this.frames.map((frame) => frame.map((point) => ({ ...point })));
  }

  reset() {
    this.frames = [];
  }
}
