import type { BoxPrompt } from '../types';

/**
 * Format a bounding box's coordinates for display.
 *
 * Returns the four rounded integer coordinates that should be shown in the UI.
 * This is the single source of truth for how box coordinates are displayed,
 * ensuring consistency between the internal state and the rendered output.
 *
 * Requirements: 4.5 — display bounding box pixel coordinate values
 */
export function formatBoxCoordinates(box: BoxPrompt): {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  display: string;
} {
  const x1 = Math.round(box.x1);
  const y1 = Math.round(box.y1);
  const x2 = Math.round(box.x2);
  const y2 = Math.round(box.y2);
  return {
    x1,
    y1,
    x2,
    y2,
    display: `[${x1}, ${y1}, ${x2}, ${y2}]`,
  };
}
