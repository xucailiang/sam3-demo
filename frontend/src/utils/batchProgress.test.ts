import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { validateBatchProgress } from './batchProgress';
import type { BatchProgress } from '../types';

const statusArb = fc.constantFrom('idle', 'processing', 'completed', 'error') as fc.Arbitrary<
  BatchProgress['status']
>;

/**
 * Generator for a valid BatchProgress where all constraints hold:
 * - 0 <= completed <= total
 * - currentIndex == completed
 * - if status == 'completed' then completed == total
 */
const validBatchProgressArb: fc.Arbitrary<BatchProgress> = fc
  .record({
    total: fc.integer({ min: 0, max: 1000 }),
    status: statusArb,
  })
  .chain(({ total, status }) => {
    // When completed, completed must equal total
    const completedArb =
      status === 'completed'
        ? fc.constant(total)
        : fc.integer({ min: 0, max: total });

    return completedArb.map((completed) => ({
      total,
      completed,
      currentIndex: completed,
      status,
    }));
  });

/**
 * Property 17: 批量处理进度一致性
 * Feature: sam3-demo, Property 17: 批量处理进度一致性
 * Validates: Requirements 8.4
 */
describe('Property 17: batch progress consistency', () => {
  it('valid progress objects pass all constraints', () => {
    fc.assert(
      fc.property(validBatchProgressArb, (progress) => {
        const result = validateBatchProgress(progress);
        expect(result.valid).toBe(true);
        expect(result.completedInRange).toBe(true);
        expect(result.currentIndexMatchesCompleted).toBe(true);
        expect(result.completedEqualsTotal).toBe(true);
      }),
      { numRuns: 100 },
    );
  });

  it('detects completed out of range (completed > total)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1000 }),
        fc.integer({ min: 1, max: 500 }),
        (total, extra) => {
          const progress: BatchProgress = {
            total,
            completed: total + extra,
            currentIndex: total + extra,
            status: 'processing',
          };
          const result = validateBatchProgress(progress);
          expect(result.completedInRange).toBe(false);
          expect(result.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('detects currentIndex not matching completed', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1000 }),
        fc.integer({ min: 0, max: 999 }),
        fc.integer({ min: -100, max: 1000 }),
        (total, completed, currentIndex) => {
          fc.pre(completed <= total);
          fc.pre(currentIndex !== completed);
          const progress: BatchProgress = {
            total,
            completed,
            currentIndex,
            status: 'processing',
          };
          const result = validateBatchProgress(progress);
          expect(result.currentIndexMatchesCompleted).toBe(false);
          expect(result.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('detects completed != total when status is completed', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 1000 }),
        fc.integer({ min: 0, max: 999 }),
        (total, completed) => {
          fc.pre(completed < total);
          const progress: BatchProgress = {
            total,
            completed,
            currentIndex: completed,
            status: 'completed',
          };
          const result = validateBatchProgress(progress);
          expect(result.completedEqualsTotal).toBe(false);
          expect(result.valid).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });
});
