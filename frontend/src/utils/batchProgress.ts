import type { BatchProgress } from '../types';

/**
 * Validation result for batch progress consistency.
 *
 * Requirements: 8.4 — progress information must be consistent.
 */
export interface BatchProgressValidation {
  valid: boolean;
  completedInRange: boolean;
  currentIndexMatchesCompleted: boolean;
  completedEqualsTotal: boolean;
}

/**
 * Validate that a BatchProgress object satisfies consistency constraints.
 *
 * Property 17: 批量处理进度一致性
 * - 0 <= completed <= total
 * - currentIndex == completed
 * - When status is 'completed', completed == total
 *
 * Validates: Requirements 8.4
 */
export function validateBatchProgress(progress: BatchProgress): BatchProgressValidation {
  const completedInRange = progress.completed >= 0 && progress.completed <= progress.total;
  const currentIndexMatchesCompleted = progress.currentIndex === progress.completed;
  const completedEqualsTotal =
    progress.status !== 'completed' || progress.completed === progress.total;

  return {
    valid: completedInRange && currentIndexMatchesCompleted && completedEqualsTotal,
    completedInRange,
    currentIndexMatchesCompleted,
    completedEqualsTotal,
  };
}
