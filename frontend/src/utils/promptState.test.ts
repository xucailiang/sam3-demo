import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { clearPoints, clearBoxes } from './promptState';
import type { PointPrompt, BoxPrompt } from '../types';

// Generators
const pointArb: fc.Arbitrary<PointPrompt> = fc.record({
  x: fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
  y: fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
  label: fc.constantFrom(0 as const, 1 as const),
});

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
 * Property 6: 清除操作重置状态
 * Feature: sam3-demo, Property 6: 清除操作重置状态
 * Validates: Requirements 3.5, 4.4
 *
 * For any non-empty list of points or boxes, executing a clear operation
 * should produce an empty array.
 */
describe('Property 6: clear operations reset state to empty array', () => {
  it('clearPoints returns empty array for any non-empty point list', () => {
    fc.assert(
      fc.property(
        fc.array(pointArb, { minLength: 1, maxLength: 50 }),
        (points) => {
          const result = clearPoints(points);
          expect(result).toEqual([]);
          expect(result).toHaveLength(0);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('clearBoxes returns empty array for any non-empty box list', () => {
    fc.assert(
      fc.property(
        fc.array(boxArb, { minLength: 1, maxLength: 50 }),
        (boxes) => {
          const result = clearBoxes(boxes);
          expect(result).toEqual([]);
          expect(result).toHaveLength(0);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('clearPoints returns empty array even for empty input', () => {
    const result = clearPoints([]);
    expect(result).toEqual([]);
  });

  it('clearBoxes returns empty array even for empty input', () => {
    const result = clearBoxes([]);
    expect(result).toEqual([]);
  });
});
