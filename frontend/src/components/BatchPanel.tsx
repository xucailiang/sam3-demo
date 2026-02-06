import { useCallback, useRef, useState } from 'react';
import type { BatchProgress, BatchResult, SegmentationResult } from '../types';
import { validateFileExtension } from '../utils/validation';
import { useBatchSegmentation } from '../hooks/useBatchSegmentation';
import { exportBatchAsZIP } from '../utils/export';

export interface BatchPanelProps {
  /** Callback when user clicks a thumbnail to view detailed result */
  onViewResult?: (result: SegmentationResult, file: File) => void;
}

/**
 * BatchPanel provides batch file upload, unified prompt input,
 * progress display, thumbnail grid, and batch export.
 *
 * Requirements: 8.1, 8.4, 8.5, 8.6, 8.7
 */
export function BatchPanel({ onViewResult }: BatchPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [textPrompt, setTextPrompt] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);

  const {
    results,
    progress,
    isProcessing,
    batchSegmentWithText,
    cancelBatch,
    clearResults,
  } = useBatchSegmentation();

  // Handle multi-file selection
  const handleFilesChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files ?? []);
      const invalid = selected.filter((f) => !validateFileExtension(f.name));
      if (invalid.length > 0) {
        setUploadError(
          `不支持的文件: ${invalid.map((f) => f.name).join(', ')}`,
        );
        return;
      }
      setUploadError(null);
      setFiles(selected);
      clearResults();
    },
    [clearResults],
  );

  // Submit batch
  const handleSubmit = useCallback(async () => {
    if (files.length === 0 || !textPrompt.trim()) return;
    await batchSegmentWithText(files, textPrompt);
  }, [files, textPrompt, batchSegmentWithText]);

  const canSubmit = files.length > 0 && !isProcessing && textPrompt.trim().length > 0;

  return (
    <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <h3 style={{ margin: 0, fontSize: '1rem' }}>批量处理</h3>

      {/* File upload */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={handleFilesChange}
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isProcessing}
          style={{
            padding: '0.35rem 0.7rem',
            borderRadius: '4px',
            border: '1px solid #555',
            background: 'transparent',
            color: 'inherit',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
            fontSize: '0.85rem',
          }}
        >
          选择图像文件
        </button>
        {files.length > 0 && (
          <span style={{ fontSize: '0.85rem', color: '#ccc' }}>
            已选择 {files.length} 个文件
          </span>
        )}
        {uploadError && (
          <span style={{ fontSize: '0.85rem', color: '#FF6B6B' }}>{uploadError}</span>
        )}
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div style={{ fontSize: '0.8rem', color: '#aaa', maxHeight: 80, overflowY: 'auto' }}>
          {files.map((f, i) => (
            <div key={i}>{f.name}</div>
          ))}
        </div>
      )}

      {/* Prompt input */}
      <input
        type="text"
        value={textPrompt}
        onChange={(e) => setTextPrompt(e.target.value)}
        placeholder="输入文本提示，多个用逗号分隔"
        disabled={isProcessing}
        style={{
          padding: '0.5rem',
          borderRadius: '4px',
          border: '1px solid #555',
          background: '#333',
          color: '#fff',
          fontSize: '0.9rem',
        }}
      />

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            border: 'none',
            background: canSubmit ? '#4ECDC4' : '#555',
            color: '#000',
            cursor: canSubmit ? 'pointer' : 'not-allowed',
            fontWeight: 600,
            fontSize: '0.9rem',
          }}
        >
          {isProcessing ? '处理中...' : '批量分割'}
        </button>
        {isProcessing && (
          <button
            onClick={cancelBatch}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              border: '1px solid #FF6B6B',
              background: 'transparent',
              color: '#FF6B6B',
              cursor: 'pointer',
              fontSize: '0.9rem',
            }}
          >
            取消
          </button>
        )}
      </div>

      {/* Progress bar */}
      {progress.status !== 'idle' && (
        <ProgressBar progress={progress} />
      )}

      {/* Results thumbnail grid */}
      {results.length > 0 && (
        <>
          <ThumbnailGrid results={results} onViewResult={onViewResult} />
          <button
            onClick={() => exportBatchAsZIP(results)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              border: '1px solid #555',
              background: 'transparent',
              color: 'inherit',
              cursor: 'pointer',
              fontSize: '0.85rem',
              alignSelf: 'flex-start',
            }}
          >
            批量导出 ZIP
          </button>
        </>
      )}
    </div>
  );
}


/** Progress bar sub-component. Requirements: 8.4 */
function ProgressBar({ progress }: { progress: BatchProgress }) {
  const pct =
    progress.total > 0
      ? Math.round((progress.completed / progress.total) * 100)
      : 0;

  const statusText =
    progress.status === 'processing'
      ? `处理中 ${progress.completed}/${progress.total}`
      : progress.status === 'completed'
        ? '处理完成'
        : progress.status === 'error'
          ? '处理出错'
          : '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: '#444',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background:
              progress.status === 'error' ? '#FF6B6B' : '#4ECDC4',
            transition: 'width 0.3s',
          }}
        />
      </div>
      <span style={{ fontSize: '0.8rem', color: '#aaa' }}>{statusText}</span>
    </div>
  );
}

/** Thumbnail grid for batch results. Requirements: 8.5, 8.6 */
function ThumbnailGrid({
  results,
  onViewResult,
}: {
  results: BatchResult[];
  onViewResult?: (result: SegmentationResult, file: File) => void;
}) {
  return (
    <div>
      <h4 style={{ margin: '0.5rem 0 0.25rem', fontSize: '0.9rem' }}>
        分割结果 ({results.filter((r) => r.result !== null).length}/{results.length})
      </h4>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(70px, 1fr))',
          gap: '0.5rem',
          maxHeight: 240,
          overflowY: 'auto',
        }}
      >
        {results.map((r, i) => (
          <Thumbnail key={i} batchResult={r} index={i} onViewResult={onViewResult} />
        ))}
      </div>
    </div>
  );
}

/** Single thumbnail. Requirements: 8.5, 8.6 */
function Thumbnail({
  batchResult,
  onViewResult,
}: {
  batchResult: BatchResult;
  index: number;
  onViewResult?: (result: SegmentationResult, file: File) => void;
}) {
  const [thumbUrl] = useState(() => URL.createObjectURL(batchResult.file));

  const handleClick = () => {
    if (batchResult.result && onViewResult) {
      onViewResult(batchResult.result, batchResult.file);
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        cursor: batchResult.result ? 'pointer' : 'default',
        border: batchResult.error
          ? '2px solid #FF6B6B'
          : batchResult.result
            ? '2px solid #4ECDC4'
            : '1px solid #555',
        borderRadius: 4,
        overflow: 'hidden',
        position: 'relative',
      }}
      title={batchResult.error ?? batchResult.file.name}
    >
      <img
        src={thumbUrl}
        alt={batchResult.file.name}
        style={{ width: '100%', height: 60, objectFit: 'cover', display: 'block' }}
      />
      {batchResult.result && (
        <span
          style={{
            position: 'absolute',
            bottom: 2,
            right: 2,
            background: 'rgba(0,0,0,0.7)',
            color: '#4ECDC4',
            fontSize: '0.65rem',
            padding: '1px 3px',
            borderRadius: 2,
          }}
        >
          {batchResult.result.count}
        </span>
      )}
      {batchResult.error && (
        <span
          style={{
            position: 'absolute',
            top: 2,
            right: 2,
            color: '#FF6B6B',
            fontSize: '0.7rem',
          }}
        >
          ✕
        </span>
      )}
    </div>
  );
}
