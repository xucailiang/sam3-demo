/**
 * Convert canvas-relative coordinates to image pixel coordinates.
 *
 * When an image is rendered on a canvas at a scaled size, a click at canvas
 * position (cx, cy) corresponds to image pixel (cx * scaleX, cy * scaleY).
 */
export function canvasToImage(
  cx: number,
  cy: number,
  scaleX: number,
  scaleY: number,
): { x: number; y: number } {
  return { x: cx * scaleX, y: cy * scaleY };
}

/**
 * Convert image pixel coordinates to canvas coordinates.
 *
 * Inverse of canvasToImage.
 */
export function imageToCanvas(
  ix: number,
  iy: number,
  scaleX: number,
  scaleY: number,
): { x: number; y: number } {
  return { x: ix / scaleX, y: iy / scaleY };
}

/**
 * Compute the scale factors from canvas coordinates to image coordinates.
 *
 * scaleX = imageWidth / canvasWidth
 * scaleY = imageHeight / canvasHeight
 */
export function computeScaleFactors(
  imageWidth: number,
  imageHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): { scaleX: number; scaleY: number } {
  return {
    scaleX: imageWidth / canvasWidth,
    scaleY: imageHeight / canvasHeight,
  };
}

/**
 * Calculate scaled image dimensions that fit within a canvas while preserving aspect ratio.
 */
export function calculateScaledSize(
  width: number,
  height: number,
  canvasWidth: number,
  canvasHeight: number
): { scaledWidth: number; scaledHeight: number } {
  const scale = Math.min(canvasWidth / width, canvasHeight / height);
  return {
    scaledWidth: Math.max(1, Math.round(width * scale)),
    scaledHeight: Math.max(1, Math.round(height * scale)),
  };
}

/**
 * Normalize bounding box coordinates from a drag interaction.
 *
 * Given two arbitrary drag points (start and end) in image space and the image
 * dimensions, produce a normalized BoxPrompt where:
 * - x1 <= x2 and y1 <= y2
 * - All coordinates are clamped to [0, imageWidth] / [0, imageHeight]
 *
 * Requirements: 4.2
 */
export function normalizeBoundingBox(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  imageWidth: number,
  imageHeight: number,
): { x1: number; y1: number; x2: number; y2: number } {
  // First normalize ordering
  const minX = Math.min(startX, endX);
  const minY = Math.min(startY, endY);
  const maxX = Math.max(startX, endX);
  const maxY = Math.max(startY, endY);

  // Then clamp to image bounds
  const x1 = Math.max(0, Math.min(minX, imageWidth));
  const y1 = Math.max(0, Math.min(minY, imageHeight));
  const x2 = Math.max(0, Math.min(maxX, imageWidth));
  const y2 = Math.max(0, Math.min(maxY, imageHeight));

  return { x1, y1, x2, y2 };
}

/**
 * Predefined palette for mask colors — visually distinct, semi-transparent friendly.
 */
const COLOR_PALETTE = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
  '#F0B27A', '#82E0AA', '#F1948A', '#AED6F1', '#D7BDE2',
  '#A3E4D7', '#FAD7A0', '#A9CCE3', '#D5F5E3', '#FADBD8',
];

/**
 * Assign a unique color to each mask. Colors cycle through the palette
 * and are guaranteed unique for up to `COLOR_PALETTE.length` masks.
 * Beyond that, HSL-generated colors fill in.
 */
export function assignMaskColors(count: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < count; i++) {
    if (i < COLOR_PALETTE.length) {
      colors.push(COLOR_PALETTE[i]);
    } else {
      // Generate additional distinct colors via evenly-spaced hues
      const hue = ((i - COLOR_PALETTE.length) * 137.508) % 360; // golden angle
      colors.push(`hsl(${Math.round(hue)}, 70%, 60%)`);
    }
  }
  return colors;
}
