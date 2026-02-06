import type { SegmentationResult } from '../types';

/**
 * Validate that a segmentation result has complete and consistent statistics.
 *
 * Requirements: 5.5 — statistics must include object count and processing time.
 *
 * Checks:
 * - count matches the masks array length
 * - processingTimeMs is non-negative
 */
export function validateStatistics(result: SegmentationResult): {
  valid: boolean;
  countMatchesMasks: boolean;
  processingTimeNonNegative: boolean;
} {
  const countMatchesMasks = result.count === result.masks.length;
  const processingTimeNonNegative = result.processingTimeMs >= 0;

  return {
    valid: countMatchesMasks && processingTimeNonNegative,
    countMatchesMasks,
    processingTimeNonNegative,
  };
}
