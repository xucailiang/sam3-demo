import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  getMaskDisplayLabel,
  getMaskDisplayConfidence,
  getMaskDisplayEntries,
} from './resultDisplay';
import type { MaskData } from '../types';

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
  label: fc.option(fc.string({ minLength: 1, maxLength: 30 }), { nil: undefined }),
  area: fc.nat({ max: 1000000 }),
  color: fc.string({ minLength: 1, maxLength: 10 }),
});

/**
 * Property 4: 结果显示包含标签和置信度
 * Feature: sam3-demo, Property 4: 结果显示包含标签和置信度
 * Validates: Requirements 2.4
 *
 * For any segmentation result mask data, the rendered output must include
 * the mask's label text and confidence score.
 */
describe('Property 4: result display includes label and confidence', () => {
  it('every mask produces a non-empty label and a percentage confidence string', () => {
    fc.assert(
      fc.property(
        maskDataArb,
        fc.nat({ max: 99 }),
        (mask, index) => {
          const label = getMaskDisplayLabel(mask, index);
          const confidence = getMaskDisplayConfidence(mask);

          // Label must be a non-empty string
          expect(label.length).toBeGreaterThan(0);

          // If the mask has an explicit label, it must be used directly
          if (mask.label !== undefined) {
            expect(label).toBe(mask.label);
          } else {
            // Fallback label uses 1-based index
            expect(label).toBe(`对象 ${index + 1}`);
          }

          // Confidence must end with '%' and contain a decimal
          expect(confidence).toMatch(/^\d+\.\d%$/);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('getMaskDisplayEntries returns one entry per mask with label and confidence', () => {
    fc.assert(
      fc.property(
        fc.array(maskDataArb, { minLength: 1, maxLength: 20 }),
        (masks) => {
          const entries = getMaskDisplayEntries(masks);

          // Entry count matches mask count
          expect(entries).toHaveLength(masks.length);

          for (let i = 0; i < masks.length; i++) {
            const entry = entries[i];
            // Each entry has a non-empty label
            expect(entry.label.length).toBeGreaterThan(0);
            // Each entry has a percentage confidence
            expect(entry.confidence).toMatch(/%$/);

            // Label matches expected value
            if (masks[i].label !== undefined) {
              expect(entry.label).toBe(masks[i].label);
            } else {
              expect(entry.label).toBe(`对象 ${i + 1}`);
            }
          }
        },
      ),
      { numRuns: 100 },
    );
  });
});
