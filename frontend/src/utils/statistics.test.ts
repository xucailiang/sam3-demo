import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { validateStatistics } from './statistics';
import type { MaskData, SegmentationResult } from '../types';

// Generator for a single MaskData
const maskDataArb: fc.Arbitrary<MaskData> = fc.record({
  maskBase64: fc.string({ minLength: 1, maxLength: 20 }),
  bbox: fc.tuple(
    fc.double({ min: 0, max: 1000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 1000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 1000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 1000, noNaN: true, noDefaultInfinity: true }),
  ) as fc.Arbitrary<[number, number, number, number]>,
  score: fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
  label: fc.option(fc.string({ minLength: 1, maxLength: 10 }), { nil: undefined }),
  area: fc.nat({ max: 1000000 }),
  color: fc.string({ minLength: 1, maxLength: 10 }),
});

// Generator for a valid SegmentationResult where count === masks.length
const validResultArb: fc.Arbitrary<SegmentationResult> = fc
  .array(maskDataArb, { minLength: 0, maxLength: 20 })
  .chain((masks) =>
    fc.record({
      masks: fc.constant(masks),
      count: fc.constant(masks.length),
      processingTimeMs: fc.double({ min: 0, max: 100000, noNaN: true, noDefaultInfinity: true }),
      imageSize: fc.tuple(
        fc.integer({ min: 1, max: 10000 }),
        fc.integer({ min: 1, max: 10000 }),
      ) as fc.Arbitrary<[number, number]>,
    }),
  );

/**
 * Property 12: 统计信息完整性
 * Feature: sam3-demo, Property 12: 统计信息完整性
 * Validates: Requirements 5.5
 *
 * For any completed segmentation result, the statistics must include:
 * - Object count that matches the masks array length
 * - Processing time that is non-negative
 */
describe('Property 12: statistics completeness', () => {
  it('valid results have count matching masks.length and non-negative processingTimeMs', () => {
    fc.assert(
      fc.property(validResultArb, (result) => {
        const stats = validateStatistics(result);
        expect(stats.valid).toBe(true);
        expect(stats.countMatchesMasks).toBe(true);
        expect(stats.processingTimeNonNegative).toBe(true);
      }),
      { numRuns: 100 },
    );
  });

  it('detects count mismatch when count !== masks.length', () => {
    fc.assert(
      fc.property(
        fc.array(maskDataArb, { minLength: 1, maxLength: 20 }),
        fc.integer({ min: -10, max: 100 }),
        fc.double({ min: 0, max: 10000, noNaN: true, noDefaultInfinity: true }),
        (masks, wrongCount, time) => {
          // Only test when count is actually wrong
          fc.pre(wrongCount !== masks.length);
          const result: SegmentationResult = {
            masks,
            count: wrongCount,
            processingTimeMs: time,
            imageSize: [100, 100],
          };
          const stats = validateStatistics(result);
          expect(stats.countMatchesMasks).toBe(false);
          expect(stats.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('detects negative processing time', () => {
    fc.assert(
      fc.property(
        fc.array(maskDataArb, { minLength: 0, maxLength: 10 }),
        fc.double({ min: -100000, max: -0.001, noNaN: true, noDefaultInfinity: true }),
        (masks, negativeTime) => {
          const result: SegmentationResult = {
            masks,
            count: masks.length,
            processingTimeMs: negativeTime,
            imageSize: [100, 100],
          };
          const stats = validateStatistics(result);
          expect(stats.processingTimeNonNegative).toBe(false);
          expect(stats.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });
});
