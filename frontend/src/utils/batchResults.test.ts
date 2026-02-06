import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { mapBatchResults, validateBatchResultCount } from './batchResults';
import type { SegmentationResult, MaskData } from '../types';

const arbMask: fc.Arbitrary<MaskData> = fc.record({
  maskBase64: fc.base64String({ minLength: 4 }),
  bbox: fc.tuple(
    fc.double({ min: 0, max: 5000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 5000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 5000, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0, max: 5000, noNaN: true, noDefaultInfinity: true }),
  ) as fc.Arbitrary<[number, number, number, number]>,
  score: fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
  label: fc.option(fc.string({ minLength: 1, maxLength: 30 }), { nil: undefined }),
  area: fc.nat({ max: 1_000_000 }),
  color: fc.constant('#ff0000'),
});

const arbResult: fc.Arbitrary<SegmentationResult> = fc.record({
  masks: fc.array(arbMask, { minLength: 0, maxLength: 5 }),
  count: fc.nat({ max: 100 }),
  processingTimeMs: fc.double({ min: 0, max: 60000, noNaN: true, noDefaultInfinity: true }),
  imageSize: fc.tuple(
    fc.integer({ min: 1, max: 10000 }),
    fc.integer({ min: 1, max: 10000 }),
  ) as fc.Arbitrary<[number, number]>,
});

/** Create a minimal File-like object for testing */
function makeFile(name: string): File {
  return new File([''], name, { type: 'image/png' });
}

/**
 * Property 18: 批量结果数量一致性
 * Feature: sam3-demo, Property 18: 批量结果数量一致性
 * Validates: Requirements 8.2, 8.3
 */
describe('Property 18: batch result count consistency', () => {
  it('mapBatchResults returns exactly as many entries as input files', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 20 }),
        fc.array(arbResult, { minLength: 0, maxLength: 20 }),
        (fileCount, results) => {
          const files = Array.from({ length: fileCount }, (_, i) =>
            makeFile(`image-${i}.png`),
          );
          const mapped = mapBatchResults(files, results);
          expect(mapped).toHaveLength(fileCount);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('validateBatchResultCount reports valid when counts match', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 100 }), (n) => {
        const validation = validateBatchResultCount(n, n);
        expect(validation.valid).toBe(true);
        expect(validation.inputCount).toBe(n);
        expect(validation.resultCount).toBe(n);
      }),
      { numRuns: 100 },
    );
  });

  it('validateBatchResultCount reports invalid when counts differ', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 100 }),
        fc.integer({ min: 0, max: 100 }),
        (a, b) => {
          fc.pre(a !== b);
          const validation = validateBatchResultCount(a, b);
          expect(validation.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });
});
