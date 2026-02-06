import type { BatchResult, SegmentationResult } from '../types';

/**
 * Map batch segmentation results to BatchResult array, one entry per input file.
 *
 * Property 18: 批量结果数量一致性
 * The returned array length must always equal the input files array length.
 *
 * Validates: Requirements 8.2, 8.3
 */
export function mapBatchResults(
  files: File[],
  results: SegmentationResult[],
): BatchResult[] {
  return files.map((file, i) => ({
    file,
    result: results[i] ?? null,
    error: null,
  }));
}

/**
 * Validate that batch results count matches input files count.
 *
 * Property 18: 批量结果数量一致性
 * Validates: Requirements 8.2, 8.3
 */
export function validateBatchResultCount(
  inputCount: number,
  resultCount: number,
): { valid: boolean; inputCount: number; resultCount: number } {
  return {
    valid: inputCount === resultCount,
    inputCount,
    resultCount,
  };
}
