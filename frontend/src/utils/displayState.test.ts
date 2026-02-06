import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { toggleMaskVisibility, clampOpacity } from './displayState';

/**
 * Property 10: 掩码可见性切换
 * Feature: sam3-demo, Property 10: 掩码可见性切换
 * Validates: Requirements 5.3
 *
 * For any mask visibility state S, executing a toggle produces !S.
 */
describe('Property 10: mask visibility toggle', () => {
  it('toggling any boolean state produces its negation', () => {
    fc.assert(
      fc.property(fc.boolean(), (state) => {
        expect(toggleMaskVisibility(state)).toBe(!state);
      }),
      { numRuns: 100 },
    );
  });

  it('toggling twice returns to the original state', () => {
    fc.assert(
      fc.property(fc.boolean(), (state) => {
        expect(toggleMaskVisibility(toggleMaskVisibility(state))).toBe(state);
      }),
      { numRuns: 100 },
    );
  });
});

/**
 * Property 11: 透明度值范围约束
 * Feature: sam3-demo, Property 11: 透明度值范围约束
 * Validates: Requirements 5.4
 *
 * For any opacity slider operation, the resulting value satisfies 0 <= opacity <= 1.
 */
describe('Property 11: opacity value range constraint', () => {
  it('clamped opacity is always within [0, 1] for any input', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1000, max: 1000, noNaN: true, noDefaultInfinity: true }),
        (value) => {
          const result = clampOpacity(value);
          expect(result).toBeGreaterThanOrEqual(0);
          expect(result).toBeLessThanOrEqual(1);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('values already in [0, 1] are unchanged', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (value) => {
          expect(clampOpacity(value)).toBe(value);
        },
      ),
      { numRuns: 100 },
    );
  });
});
