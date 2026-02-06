/**
 * Utility functions for mask display state management.
 *
 * Requirements: 5.3 (mask visibility toggle), 5.4 (opacity range)
 */

/**
 * Toggle mask visibility state.
 *
 * Requirements: 5.3 — clicking the toggle button flips mask visibility.
 */
export function toggleMaskVisibility(current: boolean): boolean {
  return !current;
}

/**
 * Clamp an opacity value to the valid range [0, 1].
 *
 * Requirements: 5.4 — opacity slider values must stay within 0–1.
 */
export function clampOpacity(value: number): number {
  return Math.max(0, Math.min(1, value));
}
