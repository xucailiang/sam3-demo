import { useCallback, useState } from 'react';

/** Default confidence threshold (25%) */
const DEFAULT_THRESHOLD = 0.25;

export interface UseResultFilterOptions {
  /** Initial confidence threshold (0-1), defaults to 0.25 */
  initialThreshold?: number;
}

export interface UseResultFilterReturn {
  /** Current confidence threshold (0-1) */
  threshold: number;
  /** Set the confidence threshold */
  setThreshold: (threshold: number) => void;
  /** Selection state Map (maskIndex -> isSelected) */
  selectionMap: Map<number, boolean>;
  /** Toggle selection state for a single mask */
  toggleSelection: (index: number) => void;
  /** Select all masks in the provided indices */
  selectAll: (filteredIndices: number[]) => void;
  /** Deselect all masks */
  selectNone: () => void;
  /** Invert selection for masks in the provided indices */
  invertSelection: (filteredIndices: number[]) => void;
  /** Reset state to defaults (called when new results arrive) */
  reset: () => void;
}

/**
 * Hook that manages result filtering and selection state.
 *
 * Provides threshold-based filtering controls and mask selection management
 * for the Mode A result filtering feature.
 *
 * Requirements: 1.2, 3.2, 4.1, 4.2, 4.3, 4.4, 6.2
 */
export function useResultFilter(
  options?: UseResultFilterOptions,
): UseResultFilterReturn {
  const initialThreshold = options?.initialThreshold ?? DEFAULT_THRESHOLD;

  const [threshold, setThresholdState] = useState<number>(initialThreshold);
  const [selectionMap, setSelectionMap] = useState<Map<number, boolean>>(
    () => new Map(),
  );

  const setThreshold = useCallback((newThreshold: number) => {
    // Clamp threshold to valid range [0, 1]
    const clamped = Math.max(0, Math.min(1, newThreshold));
    setThresholdState(clamped);
    // Selection state is preserved when threshold changes (Requirement 4.4)
  }, []);

  const toggleSelection = useCallback((index: number) => {
    setSelectionMap((prev) => {
      const next = new Map(prev);
      const current = next.get(index) ?? true; // Default to selected
      next.set(index, !current);
      return next;
    });
  }, []);

  const selectAll = useCallback((filteredIndices: number[]) => {
    setSelectionMap((prev) => {
      const next = new Map(prev);
      for (const index of filteredIndices) {
        next.set(index, true);
      }
      return next;
    });
  }, []);

  const selectNone = useCallback(() => {
    setSelectionMap((prev) => {
      const next = new Map(prev);
      // Set all existing entries to false
      for (const key of next.keys()) {
        next.set(key, false);
      }
      return next;
    });
  }, []);

  const invertSelection = useCallback((filteredIndices: number[]) => {
    setSelectionMap((prev) => {
      const next = new Map(prev);
      for (const index of filteredIndices) {
        const current = next.get(index) ?? true; // Default to selected
        next.set(index, !current);
      }
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    setThresholdState(DEFAULT_THRESHOLD);
    setSelectionMap(new Map());
  }, []);

  return {
    threshold,
    setThreshold,
    selectionMap,
    toggleSelection,
    selectAll,
    selectNone,
    invertSelection,
    reset,
  };
}
