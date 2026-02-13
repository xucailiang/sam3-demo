import type { MaskData } from '../types';

/**
 * Filter statistics for display
 */
export interface FilterStats {
  /** Total number of masks */
  totalCount: number;
  /** Number of masks passing the confidence threshold */
  filteredCount: number;
  /** Number of selected masks (among those passing filter) */
  selectedCount: number;
}

/**
 * Filter masks by confidence threshold.
 *
 * Returns only masks where score >= threshold.
 *
 * Requirements: 1.2
 * Validates: Property 1 - Confidence Filtering Correctness
 *
 * @param masks - Array of mask data to filter
 * @param threshold - Confidence threshold (0-1)
 * @returns Filtered array of masks with score >= threshold
 */
export function filterByConfidence<T extends { score: number }>(
  masks: T[],
  threshold: number,
): T[] {
  return masks.filter((mask) => mask.score >= threshold);
}

/**
 * Sort masks by confidence score in descending order.
 *
 * Requirements: 2.1
 * Validates: Property 2 - Score-Based Sorting
 *
 * @param masks - Array of mask data to sort
 * @returns New array sorted by score (highest first)
 */
export function sortByConfidence<T extends { score: number }>(masks: T[]): T[] {
  return [...masks].sort((a, b) => b.score - a.score);
}

/**
 * Calculate filter statistics for display.
 *
 * Requirements: 5.1
 *
 * @param masks - Array of all mask data
 * @param threshold - Current confidence threshold
 * @param selectionMap - Map of mask index to selection state
 * @returns Statistics object with counts
 */
export function getFilterStats(
  masks: MaskData[],
  threshold: number,
  selectionMap: Map<number, boolean>,
): FilterStats {
  const totalCount = masks.length;

  // Count masks passing the threshold
  const filteredCount = masks.filter((mask) => mask.score >= threshold).length;

  // Count selected masks among those passing the filter
  let selectedCount = 0;
  masks.forEach((mask, index) => {
    if (mask.score >= threshold) {
      // Default to selected if not in map
      const isSelected = selectionMap.get(index) ?? true;
      if (isSelected) {
        selectedCount++;
      }
    }
  });

  return {
    totalCount,
    filteredCount,
    selectedCount,
  };
}

/**
 * Get masks that should be exported based on selection state.
 *
 * Returns only masks that are selected (or default selected if not in map).
 *
 * Requirements: 3.5
 * Validates: Property 4 - Export Filtering Correctness
 *
 * @param masks - Array of all mask data
 * @param selectionMap - Map of mask index to selection state
 * @returns Array of selected masks
 */
export function getExportableMasks(
  masks: MaskData[],
  selectionMap: Map<number, boolean>,
): MaskData[] {
  return masks.filter((_, index) => {
    // Default to selected if not in map
    return selectionMap.get(index) ?? true;
  });
}
