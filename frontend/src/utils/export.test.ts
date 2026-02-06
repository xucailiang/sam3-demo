import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { buildExportJSON } from './export';
import type { SegmentationResult, MaskData } from '../types';

/**
 * Property 14: JSON 导出 Round-Trip
 * Feature: sam3-demo, Property 14: JSON 导出 Round-Trip
 * Validates: Requirements 9.3
 *
 * For any valid SegmentationResult, exporting to JSON (serialize) and
 * parsing back (deserialize) should produce an equivalent data structure
 * containing the same mask contours, bounding boxes, and metadata.
 */
describe('Property 14: JSON export round-trip', () => {
  // Arbitrary for a single MaskData (without the frontend-only `color` field)
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
    masks: fc.array(arbMask, { minLength: 0, maxLength: 10 }),
    count: fc.nat({ max: 100 }),
    processingTimeMs: fc.double({ min: 0, max: 60000, noNaN: true, noDefaultInfinity: true }),
    imageSize: fc.tuple(
      fc.integer({ min: 1, max: 10000 }),
      fc.integer({ min: 1, max: 10000 }),
    ) as fc.Arbitrary<[number, number]>,
  });

  it('JSON.parse(JSON.stringify(buildExportJSON(result))) equals buildExportJSON(result)', () => {
    fc.assert(
      fc.property(arbResult, (result) => {
        const exported = buildExportJSON(result);
        const json = JSON.stringify(exported);
        const parsed = JSON.parse(json);

        // Top-level metadata
        expect(parsed.count).toBe(exported.count);
        expect(parsed.processingTimeMs).toBe(exported.processingTimeMs);
        expect(parsed.imageSize).toEqual(exported.imageSize);

        // Masks array length
        expect(parsed.masks).toHaveLength((exported as { masks: unknown[] }).masks.length);

        // Each mask preserves bbox, score, label, area, maskBase64
        for (let i = 0; i < parsed.masks.length; i++) {
          const orig = (exported as { masks: Array<Record<string, unknown>> }).masks[i];
          const rt = parsed.masks[i];
          expect(rt.bbox).toEqual(orig.bbox);
          expect(rt.score).toBe(orig.score);
          expect(rt.label).toBe(orig.label);
          expect(rt.area).toBe(orig.area);
          expect(rt.maskBase64).toBe(orig.maskBase64);
        }
      }),
      { numRuns: 100 },
    );
  });

  it('exported JSON does not include the frontend-only color field', () => {
    fc.assert(
      fc.property(arbResult, (result) => {
        const exported = buildExportJSON(result) as { masks: Array<Record<string, unknown>> };
        for (const mask of exported.masks) {
          expect(mask).not.toHaveProperty('color');
        }
      }),
      { numRuns: 100 },
    );
  });
});
