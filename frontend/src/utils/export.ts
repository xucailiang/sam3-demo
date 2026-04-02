import type { SegmentationResult, MaskData } from '../types';

/**
 * Trigger a browser file download for the given blob.
 */
function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Decode a base64-encoded PNG mask into an HTMLImageElement.
 */
function decodeMaskImage(maskBase64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = `data:image/png;base64,${maskBase64}`;
  });
}

/**
 * Export a single binary mask as a PNG file.
 *
 * Renders the mask onto an offscreen canvas and triggers a download.
 *
 * Requirements: 9.1
 */
export async function exportMaskAsPNG(
  mask: MaskData,
  imageSize: [number, number],
  filename = 'mask.png',
): Promise<void> {
  const [width, height] = imageSize;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const maskImg = await decodeMaskImage(mask.maskBase64);
  ctx.drawImage(maskImg, 0, 0, width, height);

  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, filename);
  }, 'image/png');
}

/**
 * Export the original image with mask overlays as a PNG file.
 *
 * Draws the source image first, then composites each mask with its assigned
 * color at the given opacity.
 *
 * Requirements: 9.2
 */
export async function exportOverlayAsPNG(
  image: HTMLImageElement,
  masks: MaskData[],
  opacity: number,
  filename = 'overlay.png',
): Promise<void> {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Draw original image
  ctx.drawImage(image, 0, 0);

  // Overlay each mask
  for (const mask of masks) {
    const maskImg = await decodeMaskImage(mask.maskBase64);

    // Create a temporary canvas to colorize the mask
    const tmp = document.createElement('canvas');
    tmp.width = canvas.width;
    tmp.height = canvas.height;
    const tmpCtx = tmp.getContext('2d');
    if (!tmpCtx) continue;

    // Draw the mask
    tmpCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);

    // Use source-in to tint the mask with the assigned color
    tmpCtx.globalCompositeOperation = 'source-in';
    tmpCtx.fillStyle = mask.color;
    tmpCtx.fillRect(0, 0, canvas.width, canvas.height);

    // Composite onto the main canvas with the desired opacity
    ctx.globalAlpha = opacity;
    ctx.drawImage(tmp, 0, 0);
    ctx.globalAlpha = 1.0;
  }

  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, filename);
  }, 'image/png');
}

/**
 * Build a plain JSON-serializable object from a SegmentationResult.
 *
 * Includes mask contours (bbox), scores, labels, categories, and metadata.
 * The maskBase64 field is preserved so the data can be round-tripped.
 *
 * Requirements: 9.3, 6.1
 */
export function buildExportJSON(result: SegmentationResult): Record<string, unknown> {
  // Group masks by category for statistics
  const categoryStats: Record<string, number> = {};
  for (const m of result.masks) {
    const cat = m.category ?? '未分类';
    categoryStats[cat] = (categoryStats[cat] ?? 0) + 1;
  }
  
  return {
    count: result.count,
    processingTimeMs: result.processingTimeMs,
    imageSize: result.imageSize,
    // Category statistics - Requirement 6.1
    categoryStats,
    masks: result.masks.map((m) => ({
      bbox: m.bbox,
      score: m.score,
      label: m.label ?? null,
      // Include category in export - Requirement 6.1
      category: m.category ?? null,
      area: m.area,
      maskBase64: m.maskBase64,
    })),
  };
}

/**
 * Export the segmentation result as a JSON file download.
 *
 * Requirements: 9.3
 */
export function exportResultAsJSON(
  result: SegmentationResult,
  filename = 'result.json',
): void {
  const data = buildExportJSON(result);
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  downloadBlob(blob, filename);
}


import { zipSync, strToU8 } from 'fflate';
import type { BatchResult } from '../types';

/**
 * Export all batch segmentation results as a ZIP file.
 *
 * Each image's results are placed in a subfolder named after the file.
 * Contains the JSON result for each image with category information.
 * Includes a summary.json with overall statistics.
 *
 * Requirements: 8.7, 6.1
 */
export function exportBatchAsZIP(
  results: BatchResult[],
  filename = 'batch_results.zip',
): void {
  const files: Record<string, Uint8Array> = {};
  
  // Track overall category statistics
  const overallCategoryStats: Record<string, number> = {};
  let totalDetections = 0;
  let successCount = 0;
  let failedCount = 0;

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const baseName = r.file.name.replace(/\.[^.]+$/, '');
    const prefix = `${String(i + 1).padStart(3, '0')}_${baseName}`;

    if (r.result) {
      successCount++;
      totalDetections += r.result.count;
      
      // Aggregate category statistics
      for (const m of r.result.masks) {
        const cat = m.category ?? '未分类';
        overallCategoryStats[cat] = (overallCategoryStats[cat] ?? 0) + 1;
      }
      
      const jsonData = buildExportJSON(r.result);
      const jsonStr = JSON.stringify(jsonData, null, 2);
      files[`${prefix}/result.json`] = strToU8(jsonStr);
    } else if (r.error) {
      failedCount++;
      files[`${prefix}/error.txt`] = strToU8(r.error);
    }
  }
  
  // Create summary.json with overall statistics - Requirement 6.1
  const summary = {
    totalImages: results.length,
    successCount,
    failedCount,
    totalDetections,
    categoryStats: overallCategoryStats,
    exportedAt: new Date().toISOString(),
  };
  files['summary.json'] = strToU8(JSON.stringify(summary, null, 2));

  const zipped = zipSync(files);
  const blob = new Blob([zipped], { type: 'application/zip' });
  downloadBlob(blob, filename);
}
