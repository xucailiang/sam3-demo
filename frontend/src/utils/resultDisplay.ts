import type { MaskData } from '../types';

/**
 * Derive the display label for a mask.
 *
 * Requirements: 2.4 — each detected object must show its label.
 * Falls back to a 1-based index label when no label is provided.
 */
export function getMaskDisplayLabel(mask: MaskData, index: number): string {
  return mask.label ?? `对象 ${index + 1}`;
}

/**
 * Derive the display confidence string for a mask.
 *
 * Requirements: 2.4 — each detected object must show its confidence score.
 * Formats as a percentage with one decimal place.
 */
export function getMaskDisplayConfidence(mask: MaskData): string {
  return `${(mask.score * 100).toFixed(1)}%`;
}

/**
 * For a list of masks, return the display entries (label + confidence text).
 *
 * Requirements: 2.4 — result display must include label and confidence for every mask.
 */
export function getMaskDisplayEntries(
  masks: MaskData[],
): Array<{ label: string; confidence: string }> {
  return masks.map((m, i) => ({
    label: getMaskDisplayLabel(m, i),
    confidence: getMaskDisplayConfidence(m),
  }));
}
