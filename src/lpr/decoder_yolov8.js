// Decodes YOLOv8 detection output tensor:
// Standard (5 channels): [1, 5, num_anchors] -> cx, cy, w, h, score
// OBB (6 channels):      [1, 6, num_anchors] -> cx, cy, w, h, score, angle (radians)

import { obbToQuad, nmsOBB } from './nms.js';

function nmsAABB(candidates, iouThresh = 0.4) {
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const cand of candidates) {
    let overlap = false;
    for (const k of kept) {
      const xx1 = Math.max(cand.left, k.left);
      const yy1 = Math.max(cand.top, k.top);
      const xx2 = Math.min(cand.right, k.right);
      const yy2 = Math.min(cand.bottom, k.bottom);
      const interW = Math.max(0, xx2 - xx1);
      const interH = Math.max(0, yy2 - yy1);
      const inter = interW * interH;
      const union = (cand.w * cand.h) + (k.w * k.h) - inter;
      if (union > 0 && inter / union > iouThresh) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      kept.push(cand);
    }
  }
  return kept;
}

/**
 * Decodes YOLOv8 / YOLOv8-OBB output tensor and returns NMS-filtered detections.
 * @param {object} outputTensor ORT tensor for output0 [1, C, N]
 * @param {number} scoreThresh Minimum detection confidence
 * @param {number} iouThresh NMS IoU threshold
 * @param {number} topK Maximum detections to keep
 */
export function decodeYolov8(outputTensor, scoreThresh = 0.25, iouThresh = 0.40, topK = 50) {
  if (!outputTensor || !outputTensor.data) return [];
  const dims = outputTensor.dims || [];
  const d = outputTensor.data;

  // Determine channels and anchors from dimensions
  let numChannels = 5;
  let numAnchors = 0;
  if (dims.length === 3) {
    numChannels = dims[1];
    numAnchors = dims[2];
  } else if (dims.length === 2) {
    numChannels = dims[0];
    numAnchors = dims[1];
  } else {
    numChannels = (d.length % 6 === 0) ? 6 : 5;
    numAnchors = Math.floor(d.length / numChannels);
  }

  const isObb = numChannels >= 6;
  const candidates = [];
  const plane = numAnchors;

  for (let i = 0; i < numAnchors; i++) {
    const score = d[4 * plane + i];
    if (score >= scoreThresh) {
      const cx = d[0 * plane + i];
      const cy = d[1 * plane + i];
      const w = d[2 * plane + i];
      const h = d[3 * plane + i];
      const angle = isObb ? d[5 * plane + i] : 0;

      let quad;
      let left, top, right, bottom;

      if (isObb) {
        quad = obbToQuad(cx, cy, w, h, angle);
        const xs = [quad[0][0], quad[1][0], quad[2][0], quad[3][0]];
        const ys = [quad[0][1], quad[1][1], quad[2][1], quad[3][1]];
        left = Math.min(...xs);
        top = Math.min(...ys);
        right = Math.max(...xs);
        bottom = Math.max(...ys);
      } else {
        left = cx - w / 2;
        top = cy - h / 2;
        right = cx + w / 2;
        bottom = cy + h / 2;
        quad = [
          [left, top],
          [right, top],
          [right, bottom],
          [left, bottom]
        ];
      }

      candidates.push({
        cx,
        cy,
        w,
        h,
        angle,
        score,
        left,
        top,
        right,
        bottom,
        quad
      });
    }
  }

  const kept = isObb ? nmsOBB(candidates, iouThresh) : nmsAABB(candidates, iouThresh);
  return kept.slice(0, topK);
}
