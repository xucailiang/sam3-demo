import type { PointPrompt, BoxPrompt } from '../types';

/**
 * Clear all point prompts, returning an empty array.
 *
 * Requirements: 3.5 — "清除点"按钮移除所有 Point_Prompt
 */
export function clearPoints(_points: PointPrompt[]): PointPrompt[] {
  return [];
}

/**
 * Clear all box prompts, returning an empty array.
 *
 * Requirements: 4.4 — "清除框"按钮移除所有 Box_Prompt
 */
export function clearBoxes(_boxes: BoxPrompt[]): BoxPrompt[] {
  return [];
}
