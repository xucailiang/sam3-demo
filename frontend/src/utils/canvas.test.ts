import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { calculateScaledSize, assignMaskColors, canvasToImage, imageToCanvas, computeScaleFactors, normalizeBoundingBox } from './canvas';

describe('calculateScaledSize', () => {
  it('scales down a landscape image to fit canvas', () => {
    const { scaledWidth, scaledHeight } = calculateScaledSize(2000, 1000, 800, 600);
    expect(scaledWidth).toBeLessThanOrEqual(800);
    expect(scaledHeight).toBeLessThanOrEqual(600);
  });

  it('scales down a portrait image to fit canvas', () => {
    const { scaledWidth, scaledHeight } = calculateScaledSize(1000, 2000, 800, 600);
    expect(scaledWidth).toBeLessThanOrEqual(800);
    expect(scaledHeight).toBeLessThanOrEqual(600);
  });

  it('preserves aspect ratio', () => {
    const { scaledWidth, scaledHeight } = calculateScaledSize(1600, 900, 800, 600);
    const originalRatio = 1600 / 900;
    const scaledRatio = scaledWidth / scaledHeight;
    expect(Math.abs(originalRatio - scaledRatio)).toBeLessThan(0.02);
  });

  it('handles image smaller than canvas', () => {
    const { scaledWidth, scaledHeight } = calculateScaledSize(400, 300, 800, 600);
    // scale = min(2, 2) = 2 → 800x600
    expect(scaledWidth).toBeLessThanOrEqual(800);
    expect(scaledHeight).toBeLessThanOrEqual(600);
  });

  it('handles square image', () => {
    const { scaledWidth, scaledHeight } = calculateScaledSize(500, 500, 800, 600);
    expect(scaledWidth).toBe(scaledHeight);
    expect(scaledWidth).toBeLessThanOrEqual(800);
    expect(scaledHeight).toBeLessThanOrEqual(600);
  });
});

/**
 * Property 2: 图像缩放保持宽高比
 * Feature: sam3-demo, Property 2: 图像缩放保持宽高比
 * Validates: Requirements 1.5
 */
describe('Property 2: calculateScaledSize preserves aspect ratio', () => {
  const dimension = fc.integer({ min: 1, max: 10000 });

  it('scaled dimensions do not exceed canvas bounds', () => {
    fc.assert(
      fc.property(dimension, dimension, dimension, dimension,
        (width, height, canvasWidth, canvasHeight) => {
          const { scaledWidth, scaledHeight } = calculateScaledSize(width, height, canvasWidth, canvasHeight);
          expect(scaledWidth).toBeLessThanOrEqual(canvasWidth);
          expect(scaledHeight).toBeLessThanOrEqual(canvasHeight);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('aspect ratio is preserved within rounding tolerance', () => {
    const dim = fc.integer({ min: 2, max: 10000 });
    fc.assert(
      fc.property(dim, dim, dim, dim,
        (width, height, canvasWidth, canvasHeight) => {
          const { scaledWidth, scaledHeight } = calculateScaledSize(width, height, canvasWidth, canvasHeight);
          if (scaledWidth <= 1 || scaledHeight <= 1) return;
          const originalRatio = width / height;
          const scaledRatio = scaledWidth / scaledHeight;
          // Math.round introduces ±0.5 error on each dimension.
          // Compute the maximum possible ratio from rounding: (sw+0.5)/(sh-0.5)
          // and minimum: (sw-0.5)/(sh+0.5), where sw/sh are the true (unrounded) values.
          // Since we only have the rounded values, the true ratio W/H = width/height,
          // and the rounded ratio can deviate. A safe upper bound on the deviation:
          const maxRatio = (scaledWidth + 0.5) / (scaledHeight - 0.5);
          const minRatio = (scaledWidth - 0.5) / (scaledHeight + 0.5);
          // The original ratio should fall within this interval (or very close)
          // and the scaled ratio should also be within this interval.
          // So the difference between originalRatio and scaledRatio should be bounded
          // by the width of this interval.
          const tolerance = maxRatio - minRatio;
          expect(Math.abs(originalRatio - scaledRatio)).toBeLessThanOrEqual(tolerance);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('scaled dimensions are positive integers', () => {
    fc.assert(
      fc.property(dimension, dimension, dimension, dimension,
        (width, height, canvasWidth, canvasHeight) => {
          const { scaledWidth, scaledHeight } = calculateScaledSize(width, height, canvasWidth, canvasHeight);
          expect(scaledWidth).toBeGreaterThan(0);
          expect(scaledHeight).toBeGreaterThan(0);
          expect(Number.isInteger(scaledWidth)).toBe(true);
          expect(Number.isInteger(scaledHeight)).toBe(true);
        }
      ),
      { numRuns: 100 }
    );
  });
});

describe('assignMaskColors', () => {
  it('returns empty array for 0 masks', () => {
    expect(assignMaskColors(0)).toEqual([]);
  });

  it('returns correct number of colors', () => {
    expect(assignMaskColors(5)).toHaveLength(5);
    expect(assignMaskColors(25)).toHaveLength(25);
  });

  it('returns unique colors for up to palette size', () => {
    const colors = assignMaskColors(10);
    const unique = new Set(colors);
    expect(unique.size).toBe(10);
  });

  it('returns unique colors beyond palette size', () => {
    const colors = assignMaskColors(25);
    const unique = new Set(colors);
    expect(unique.size).toBe(25);
  });
});

/**
 * Property 9: 掩码颜色唯一性
 * Feature: sam3-demo, Property 9: 掩码颜色唯一性
 * Validates: Requirements 5.1
 */
describe('Property 9: assignMaskColors returns unique colors', () => {
  it('all assigned colors are pairwise distinct for any count > 1', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 100 }),
        (count) => {
          const colors = assignMaskColors(count);
          expect(colors).toHaveLength(count);
          const unique = new Set(colors);
          expect(unique.size).toBe(count);
        }
      ),
      { numRuns: 100 }
    );
  });
});


/**
 * Property 5: 点击坐标记录
 * Feature: sam3-demo, Property 5: 点击坐标记录
 * Validates: Requirements 3.1
 *
 * For any canvas click position, the coordinate conversion to image space
 * and back to canvas space should produce the original position (round-trip),
 * and the converted image coordinates should be consistent with the scale factors.
 */
describe('Property 5: click coordinate recording — canvasToImage / imageToCanvas round-trip', () => {
  // Positive floats for coordinates and dimensions
  const posFloat = fc.double({ min: 1, max: 10000, noNaN: true, noDefaultInfinity: true });
  const coordFloat = fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true });

  it('canvasToImage → imageToCanvas round-trip preserves coordinates', () => {
    fc.assert(
      fc.property(
        coordFloat, coordFloat, posFloat, posFloat,
        (cx, cy, scaleX, scaleY) => {
          const imgPos = canvasToImage(cx, cy, scaleX, scaleY);
          const backToCanvas = imageToCanvas(imgPos.x, imgPos.y, scaleX, scaleY);
          // Floating-point round-trip should be very close to original
          expect(backToCanvas.x).toBeCloseTo(cx, 8);
          expect(backToCanvas.y).toBeCloseTo(cy, 8);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('imageToCanvas → canvasToImage round-trip preserves coordinates', () => {
    fc.assert(
      fc.property(
        coordFloat, coordFloat, posFloat, posFloat,
        (ix, iy, scaleX, scaleY) => {
          const canvasPos = imageToCanvas(ix, iy, scaleX, scaleY);
          const backToImage = canvasToImage(canvasPos.x, canvasPos.y, scaleX, scaleY);
          expect(backToImage.x).toBeCloseTo(ix, 8);
          expect(backToImage.y).toBeCloseTo(iy, 8);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('canvasToImage produces coordinates proportional to scale factors', () => {
    fc.assert(
      fc.property(
        coordFloat, coordFloat, posFloat, posFloat,
        (cx, cy, scaleX, scaleY) => {
          const imgPos = canvasToImage(cx, cy, scaleX, scaleY);
          expect(imgPos.x).toBeCloseTo(cx * scaleX, 8);
          expect(imgPos.y).toBeCloseTo(cy * scaleY, 8);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('computeScaleFactors produces correct scale values for calculateScaledSize output', () => {
    const dim = fc.integer({ min: 1, max: 10000 });
    fc.assert(
      fc.property(
        dim, dim, dim, dim,
        (imgW, imgH, canvasW, canvasH) => {
          const { scaledWidth, scaledHeight } = calculateScaledSize(imgW, imgH, canvasW, canvasH);
          const { scaleX, scaleY } = computeScaleFactors(imgW, imgH, scaledWidth, scaledHeight);
          // A canvas click at (scaledWidth, scaledHeight) should map to (imgW, imgH)
          const imgPos = canvasToImage(scaledWidth, scaledHeight, scaleX, scaleY);
          expect(imgPos.x).toBeCloseTo(imgW, 5);
          expect(imgPos.y).toBeCloseTo(imgH, 5);
        },
      ),
      { numRuns: 100 },
    );
  });
});


/**
 * Property 7: 边界框坐标记录
 * Feature: sam3-demo, Property 7: 边界框坐标记录
 * Validates: Requirements 4.2
 *
 * For any drag start/end coordinates, the normalized bounding box should satisfy:
 * - x1 <= x2 and y1 <= y2 (coordinates are normalized)
 * - All coordinates are within image bounds [0, imageWidth] x [0, imageHeight]
 */
describe('Property 7: bounding box coordinate normalization', () => {
  const coord = fc.double({ min: -500, max: 10000, noNaN: true, noDefaultInfinity: true });
  const dim = fc.double({ min: 1, max: 10000, noNaN: true, noDefaultInfinity: true });

  it('normalized box always has x1 <= x2 and y1 <= y2', () => {
    fc.assert(
      fc.property(
        coord, coord, coord, coord, dim, dim,
        (startX, startY, endX, endY, imageWidth, imageHeight) => {
          const box = normalizeBoundingBox(startX, startY, endX, endY, imageWidth, imageHeight);
          expect(box.x1).toBeLessThanOrEqual(box.x2);
          expect(box.y1).toBeLessThanOrEqual(box.y2);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('normalized box coordinates are clamped within image bounds', () => {
    fc.assert(
      fc.property(
        coord, coord, coord, coord, dim, dim,
        (startX, startY, endX, endY, imageWidth, imageHeight) => {
          const box = normalizeBoundingBox(startX, startY, endX, endY, imageWidth, imageHeight);
          expect(box.x1).toBeGreaterThanOrEqual(0);
          expect(box.y1).toBeGreaterThanOrEqual(0);
          expect(box.x2).toBeLessThanOrEqual(imageWidth);
          expect(box.y2).toBeLessThanOrEqual(imageHeight);
        },
      ),
      { numRuns: 100 },
    );
  });
});
