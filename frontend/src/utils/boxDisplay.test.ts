import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { formatBoxCoordinates } from './boxDisplay';
import type { BoxPrompt } from '../types';

/**
 * Arbitrary generator for valid BoxPrompt values (x1 <= x2, y1 <= y2).
 */
const boxArb: fc.Arbitrary<BoxPrompt> = fc
  .tuple(
    fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([a, b, c, d]) => ({
    x1: Math.min(a, c),
    y1: Math.min(b, d),
    x2: Math.max(a, c),
    y2: Math.max(b, d),
  }));

/**
 * Property 8: 边界框坐标显示一致性
 * Feature: sam3-demo, Property 8: 边界框坐标显示一致性
 * Validates: Requirements 4.5
 *
 * For any recorded bounding box, the displayed coordinate values must be
 * consistent with the internal state (rounded to integers).
 */
describe('Property 8: bounding box coordinate display consistency', () => {
  it('displayed coordinates equal Math.round of internal state', () => {
    fc.assert(
      fc.property(boxArb, (box) => {
        const result = formatBoxCoordinates(box);
        expect(result.x1).toBe(Math.round(box.x1));
        expect(result.y1).toBe(Math.round(box.y1));
        expect(result.x2).toBe(Math.round(box.x2));
        expect(result.y2).toBe(Math.round(box.y2));
      }),
      { numRuns: 100 },
    );
  });

  it('display string matches the formatted coordinate values', () => {
    fc.assert(
      fc.property(boxArb, (box) => {
        const result = formatBoxCoordinates(box);
        expect(result.display).toBe(
          `[${result.x1}, ${result.y1}, ${result.x2}, ${result.y2}]`,
        );
      }),
      { numRuns: 100 },
    );
  });

  it('display string is parseable back to the same rounded coordinates', () => {
    fc.assert(
      fc.property(boxArb, (box) => {
        const result = formatBoxCoordinates(box);
        // Parse the display string back
        const match = result.display.match(/^\[(-?\d+), (-?\d+), (-?\d+), (-?\d+)\]$/);
        expect(match).not.toBeNull();
        const [, px1, py1, px2, py2] = match!;
        expect(Number(px1)).toBe(result.x1);
        expect(Number(py1)).toBe(result.y1);
        expect(Number(px2)).toBe(result.x2);
        expect(Number(py2)).toBe(result.y2);
      }),
      { numRuns: 100 },
    );
  });
});
