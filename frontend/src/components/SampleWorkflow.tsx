import { useCallback, useRef, useState, useEffect } from 'react';
import type { BoxPrompt, MaskData, BatchResult, WorkflowMode } from '../types';
import { validateFileExtension } from '../utils/validation';
import { assignMaskColors, calculateScaledSize } from '../utils/canvas';
import { useBatchSegmentation } from '../hooks/useBatchSegmentation';
import { useSegmentation } from '../hooks/useSegmentation';
import { exportBatchAsZIP } from '../utils/export';
import { FilterControls } from './FilterControls';
import { sortByConfidence, filterByConfidence, getFilterStats } from '../utils/filter';

const CANVAS_MAX = 520;
const RESULT_CANVAS_MAX = 400;

/**
 * SampleWorkflow implements a step-by-step workflow with two modes:
 * - Mode A (stitch): Upload sample image + draw BBox + batch segment with stitching
 * - Mode B (text): Enter text description + batch segment with text prompts
 */
export function SampleWorkflow() {
  // === Mode selection ===
  const [mode, setMode] = useState<WorkflowMode>('stitch');

  // === Mode A: Sample image + box ===
  const [sampleFile, setSampleFile] = useState<File | null>(null);
  const [sampleImg, setSampleImg] = useState<HTMLImageElement | null>(null);
  const [sampleBox, setSampleBox] = useState<BoxPrompt | null>(null);
  const sampleInputRef = useRef<HTMLInputElement>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);

  // Sample preview segmentation
  const { result: sampleResult, isLoading: sampleLoading, error: sampleError, segmentWithBoxes, clearResult: clearSampleResult } = useSegmentation();

  // === Mode B: Text prompt ===
  const [textPrompt, setTextPrompt] = useState<string>('');

  // === Shared: Batch images ===
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const { results: batchResults, progress, isProcessing, batchSegmentWithText, batchSegmentWithStitch, cancelBatch, clearResults } = useBatchSegmentation();

  // === Step 3: View detail ===
  const [viewIndex, setViewIndex] = useState<number | null>(null);

  // === Filter and selection state ===
  const [filterThreshold, setFilterThreshold] = useState<number>(0.80);
  const [batchSelectionState, setBatchSelectionState] = useState<Map<number, Map<number, boolean>>>(new Map());

  // === Inference confidence (backend conf parameter) ===
  const [inferenceConfidence, setInferenceConfidence] = useState<number>(0.25);

  // Get selection map for current viewed image
  const currentSelectionMap = viewIndex !== null 
    ? (batchSelectionState.get(viewIndex) ?? new Map<number, boolean>())
    : new Map<number, boolean>();

  // Update selection map for current viewed image
  const handleSelectionChange = useCallback((newSelectionMap: Map<number, boolean>) => {
    if (viewIndex === null) return;
    setBatchSelectionState(prev => {
      const next = new Map(prev);
      next.set(viewIndex, newSelectionMap);
      return next;
    });
  }, [viewIndex]);

  // Upload error
  const [uploadError, setUploadError] = useState<string | null>(null);

  // --- Mode switch handler ---
  const handleModeSwitch = useCallback((newMode: WorkflowMode) => {
    if (newMode === mode) return;
    // Clear all state when switching modes
    setMode(newMode);
    // Clear Mode A state
    setSampleFile(null);
    setSampleImg(null);
    setSampleBox(null);
    setDragStart(null);
    setDragCurrent(null);
    clearSampleResult();
    // Clear Mode B state
    setTextPrompt('');
    // Clear shared state
    setBatchFiles([]);
    clearResults();
    setViewIndex(null);
    setUploadError(null);
    // Reset filter state (Requirement 6.2)
    setFilterThreshold(0.80);
    setBatchSelectionState(new Map());
  }, [mode, clearSampleResult, clearResults]);

  // --- Sample image upload ---
  const handleSampleUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!validateFileExtension(file.name)) {
      setUploadError('不支持的文件格式，请上传 jpg、jpeg、png 或 webp');
      return;
    }
    setUploadError(null);
    setSampleFile(file);
    setSampleBox(null);
    clearSampleResult();
    clearResults();
    setBatchFiles([]);
    setViewIndex(null);

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setSampleImg(img);
    img.src = url;
  }, [clearSampleResult, clearResults]);

  // --- Sample canvas dimensions ---
  const sampleScaled = sampleImg
    ? calculateScaledSize(sampleImg.naturalWidth, sampleImg.naturalHeight, CANVAS_MAX, CANVAS_MAX)
    : { scaledWidth: CANVAS_MAX, scaledHeight: 300 };
  const sW = sampleScaled.scaledWidth;
  const sH = sampleScaled.scaledHeight;
  const sScaleX = sampleImg ? sampleImg.naturalWidth / sW : 1;
  const sScaleY = sampleImg ? sampleImg.naturalHeight / sH : 1;

  // Colored masks for sample result
  const sampleMasks: MaskData[] = sampleResult
    ? (() => {
        const colors = assignMaskColors(sampleResult.masks.length);
        return sampleResult.masks.map((m, i) => ({ ...m, color: colors[i] }));
      })()
    : [];

  // --- Draw sample canvas ---
  const drawSampleCanvas = useCallback(() => {
    const canvas = sampleCanvasRef.current;
    if (!canvas || !sampleImg) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, sW, sH);
    ctx.drawImage(sampleImg, 0, 0, sW, sH);

    // Draw masks overlay
    if (sampleMasks.length > 0) {
      for (const m of sampleMasks) {
        const maskImg = new Image();
        maskImg.src = `data:image/png;base64,${m.maskBase64}`;
        if (maskImg.complete) {
          const off = document.createElement('canvas');
          off.width = sW;
          off.height = sH;
          const offCtx = off.getContext('2d');
          if (offCtx) {
            offCtx.drawImage(maskImg, 0, 0, sW, sH);
            offCtx.globalCompositeOperation = 'source-in';
            offCtx.fillStyle = m.color;
            offCtx.fillRect(0, 0, sW, sH);
            ctx.save();
            ctx.globalAlpha = 0.45;
            ctx.drawImage(off, 0, 0);
            ctx.restore();
          }
        }
      }
    }

    // Draw completed box
    if (sampleBox) {
      const x = sampleBox.x1 / sScaleX;
      const y = sampleBox.y1 / sScaleY;
      const w = (sampleBox.x2 - sampleBox.x1) / sScaleX;
      const h = (sampleBox.y2 - sampleBox.y1) / sScaleY;
      ctx.strokeStyle = '#FF6B6B';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      // Label
      ctx.fillStyle = 'rgba(255,107,107,0.85)';
      ctx.fillRect(x, y - 18, 60, 18);
      ctx.fillStyle = '#fff';
      ctx.font = '11px sans-serif';
      ctx.fillText('标记区域', x + 4, y - 5);
    }

    // Draw in-progress drag
    if (dragStart && dragCurrent) {
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(dragStart.x, dragStart.y, dragCurrent.x - dragStart.x, dragCurrent.y - dragStart.y);
      ctx.setLineDash([]);
    }
  }, [sampleImg, sW, sH, sampleBox, sScaleX, sScaleY, dragStart, dragCurrent, sampleMasks]);

  useEffect(() => { drawSampleCanvas(); }, [drawSampleCanvas]);

  // Redraw when mask images load (async)
  useEffect(() => {
    if (sampleMasks.length === 0) return;
    const imgs = sampleMasks.map((m) => {
      const img = new Image();
      img.src = `data:image/png;base64,${m.maskBase64}`;
      return img;
    });
    let loaded = 0;
    const onLoad = () => { loaded++; if (loaded === imgs.length) drawSampleCanvas(); };
    imgs.forEach((img) => { img.onload = onLoad; if (img.complete) onLoad(); });
  }, [sampleMasks, drawSampleCanvas]);

  // --- Sample canvas mouse handlers ---
  const getCanvasPos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = sampleCanvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    setDragStart(pos);
    setDragCurrent(pos);
    setSampleBox(null);
    clearSampleResult();
  }, [getCanvasPos, clearSampleResult]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragStart) setDragCurrent(getCanvasPos(e));
  }, [dragStart, getCanvasPos]);

  const handleMouseUp = useCallback(() => {
    if (!dragStart || !dragCurrent || !sampleImg) {
      setDragStart(null);
      setDragCurrent(null);
      return;
    }
    let x1 = dragStart.x * sScaleX, y1 = dragStart.y * sScaleY;
    let x2 = dragCurrent.x * sScaleX, y2 = dragCurrent.y * sScaleY;
    if (x1 > x2) [x1, x2] = [x2, x1];
    if (y1 > y2) [y1, y2] = [y2, y1];
    x1 = Math.max(0, x1); y1 = Math.max(0, y1);
    x2 = Math.min(sampleImg.naturalWidth, x2); y2 = Math.min(sampleImg.naturalHeight, y2);
    if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
      setSampleBox({ x1, y1, x2, y2 });
    }
    setDragStart(null);
    setDragCurrent(null);
  }, [dragStart, dragCurrent, sampleImg, sScaleX, sScaleY]);

  // --- Preview segmentation on sample ---
  const handlePreviewSample = useCallback(async () => {
    if (!sampleFile || !sampleBox) return;
    await segmentWithBoxes(sampleFile, [sampleBox]);
  }, [sampleFile, sampleBox, segmentWithBoxes]);

  // --- Batch upload ---
  const handleBatchUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? []);
    const invalid = selected.filter((f) => !validateFileExtension(f.name));
    if (invalid.length > 0) {
      setUploadError(`不支持的文件: ${invalid.map((f) => f.name).join(', ')}`);
      return;
    }
    setUploadError(null);
    setBatchFiles(selected);
    clearResults();
    setViewIndex(null);
  }, [clearResults]);

  // --- Batch segment ---
  const handleBatchSegment = useCallback(async () => {
    // Reset filter state when starting new batch (Requirement 6.2)
    setFilterThreshold(0.80);
    setBatchSelectionState(new Map());
    
    if (mode === 'stitch') {
      // Mode A: stitch segmentation
      if (!sampleFile || !sampleBox || batchFiles.length === 0) return;
      await batchSegmentWithStitch(batchFiles, sampleFile, sampleBox, inferenceConfidence);
    } else {
      // Mode B: text segmentation
      if (!textPrompt.trim() || batchFiles.length === 0) return;
      await batchSegmentWithText(batchFiles, textPrompt.trim());
    }
  }, [mode, sampleFile, sampleBox, batchFiles, textPrompt, batchSegmentWithStitch, batchSegmentWithText, inferenceConfidence]);

  const canBatchModeA = sampleFile !== null && sampleBox !== null && batchFiles.length > 0 && !isProcessing;
  const canBatchModeB = textPrompt.trim().length > 0 && batchFiles.length > 0 && !isProcessing;
  const canBatch = mode === 'stitch' ? canBatchModeA : canBatchModeB;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Mode Switcher */}
      <section style={sectionStyle}>
        <h3 style={{ ...stepTitleStyle, marginBottom: '0.5rem' }}>选择工作模式</h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={() => handleModeSwitch('stitch')}
            style={mode === 'stitch' ? btnModeActive : btnModeInactive}
          >
            模式 A：图像拼接 + BBox
          </button>
          <button
            onClick={() => handleModeSwitch('text')}
            style={mode === 'text' ? btnModeActive : btnModeInactive}
          >
            模式 B：纯文本
          </button>
        </div>
        <p style={{ fontSize: '0.8rem', color: '#888', margin: '0.5rem 0 0' }}>
          {mode === 'stitch'
            ? '上传样品图并标记目标区域，系统将在其他图片中找到相似对象'
            : '输入文本描述，系统将在所有图片中找到匹配的对象'}
        </p>
      </section>

      {/* Mode A: Sample Image */}
      {mode === 'stitch' && (
        <section style={sectionStyle}>
          <h3 style={stepTitleStyle}>
            <span style={stepBadgeStyle}>1</span>
            上传样品图并标记目标区域
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <input ref={sampleInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleSampleUpload} style={{ display: 'none' }} />
            <button onClick={() => sampleInputRef.current?.click()} style={btnPrimary} disabled={sampleLoading}>
              选择样品图
            </button>
            {sampleFile && <span style={{ fontSize: '0.85rem', color: '#ccc' }}>{sampleFile.name}</span>}
            {uploadError && <span style={{ fontSize: '0.85rem', color: '#FF6B6B' }}>{uploadError}</span>}
          </div>

          {sampleImg && (
            <>
              <p style={{ fontSize: '0.85rem', color: '#aaa', margin: '0 0 0.4rem' }}>
                在图像上拖拽绘制边界框，框选你想要分割的目标
              </p>
              <canvas
                ref={sampleCanvasRef}
                width={sW}
                height={sH}
                style={{ border: '1px solid #555', cursor: 'crosshair', display: 'block', borderRadius: 4 }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => { if (dragStart) { setDragStart(null); setDragCurrent(null); } }}
              />
            </>
          )}

          {sampleBox && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.85rem', color: '#ccc' }}>
                标记区域: [{Math.round(sampleBox.x1)}, {Math.round(sampleBox.y1)}, {Math.round(sampleBox.x2)}, {Math.round(sampleBox.y2)}]
              </span>
              <button onClick={() => { setSampleBox(null); clearSampleResult(); }} style={btnDanger}>重新标记</button>
              <button onClick={handlePreviewSample} disabled={sampleLoading} style={btnSecondary}>
                {sampleLoading ? '预览中...' : '预览分割效果'}
              </button>
            </div>
          )}

          {sampleError && <p style={{ color: '#FF6B6B', fontSize: '0.85rem', margin: '0.25rem 0 0' }}>{sampleError}</p>}

          {sampleResult && sampleResult.masks.length > 0 && (
            <div style={{ fontSize: '0.85rem', color: '#4ECDC4', marginTop: '0.25rem' }}>
              ✓ 在样品图中检测到 {sampleResult.count} 个对象，耗时 {sampleResult.processingTimeMs.toFixed(0)}ms
            </div>
          )}
          {sampleResult && sampleResult.masks.length === 0 && (
            <div style={{ fontSize: '0.85rem', color: '#FFEAA7', marginTop: '0.25rem' }}>
              未在标记区域中找到可分割的对象，请尝试调整框选范围
            </div>
          )}
        </section>
      )}

      {/* Mode B: Text Input */}
      {mode === 'text' && (
        <section style={sectionStyle}>
          <h3 style={stepTitleStyle}>
            <span style={stepBadgeStyle}>1</span>
            输入文本描述
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <input
              type="text"
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              placeholder="例如：cat, dog, person..."
              style={textInputStyle}
            />
            {!textPrompt.trim() && (
              <p style={{ fontSize: '0.8rem', color: '#FFEAA7', margin: 0 }}>
                请输入文本描述以启用批量分割
              </p>
            )}
          </div>
        </section>
      )}

      {/* Step 2: Batch Upload & Segment */}
      {(mode === 'text' || (mode === 'stitch' && sampleBox)) && (
        <section style={sectionStyle}>
          <h3 style={stepTitleStyle}>
            <span style={stepBadgeStyle}>2</span>
            上传目标图片，批量分割
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            <input ref={batchInputRef} type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={handleBatchUpload} style={{ display: 'none' }} />
            <button onClick={() => batchInputRef.current?.click()} disabled={isProcessing} style={btnPrimary}>
              选择图片
            </button>
            {batchFiles.length > 0 && (
              <span style={{ fontSize: '0.85rem', color: '#ccc' }}>已选择 {batchFiles.length} 张图片</span>
            )}
          </div>

          {batchFiles.length > 0 && (
            <div style={{ fontSize: '0.8rem', color: '#888', maxHeight: 60, overflowY: 'auto', marginBottom: '0.5rem' }}>
              {batchFiles.map((f, i) => <div key={i}>{f.name}</div>)}
            </div>
          )}

          {/* Inference confidence slider - only for stitch mode */}
          {mode === 'stitch' && (
            <div style={{ marginBottom: '0.75rem', padding: '0.5rem 0.75rem', background: '#1e1e1e', borderRadius: 4, border: '1px solid #444' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <label htmlFor="inference-confidence" style={{ fontSize: '0.85rem', color: '#ccc', whiteSpace: 'nowrap' }}>
                  推理置信度
                </label>
                <input
                  id="inference-confidence"
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={inferenceConfidence * 100}
                  onChange={(e) => setInferenceConfidence(Number(e.target.value) / 100)}
                  disabled={isProcessing}
                  style={{ flex: 1, minWidth: 100, maxWidth: 200 }}
                />
                <span style={{ fontSize: '0.85rem', color: '#4ECDC4', fontWeight: 600, minWidth: 45 }}>
                  {(inferenceConfidence * 100).toFixed(0)}%
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: '#888', margin: '0.35rem 0 0' }}>
                提高此值可减少返回的低置信度结果，降低可获得更多候选对象
              </p>
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button onClick={handleBatchSegment} disabled={!canBatch} style={canBatch ? btnPrimary : btnDisabled}>
              {isProcessing ? '处理中...' : '开始批量分割'}
            </button>
            {isProcessing && (
              <button onClick={cancelBatch} style={btnDanger}>取消</button>
            )}
          </div>

          {/* Progress */}
          {progress.status !== 'idle' && (
            <div style={{ marginTop: '0.5rem' }}>
              <div style={{ height: 6, borderRadius: 3, background: '#444', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${progress.total > 0 ? Math.round((progress.completed / progress.total) * 100) : 0}%`,
                  background: progress.status === 'error' ? '#FF6B6B' : '#4ECDC4',
                  transition: 'width 0.3s',
                }} />
              </div>
              <span style={{ fontSize: '0.8rem', color: '#aaa' }}>
                {progress.status === 'processing' && `处理中 ${progress.completed}/${progress.total}`}
                {progress.status === 'completed' && '处理完成'}
                {progress.status === 'error' && '处理出错'}
              </span>
            </div>
          )}
        </section>
      )}

      {/* Step 3: Results */}
      {batchResults.length > 0 && (
        <section style={sectionStyle}>
          <h3 style={stepTitleStyle}>
            <span style={stepBadgeStyle}>3</span>
            分割结果
            <span style={{ fontSize: '0.85rem', fontWeight: 400, color: '#aaa', marginLeft: '0.5rem' }}>
              ({batchResults.filter((r) => r.result && r.result.count > 0).length}/{batchResults.length} 成功)
            </span>
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: '0.75rem',
          }}>
            {batchResults.map((r, i) => (
              <ResultCard 
                key={i} 
                batchResult={r} 
                isSelected={viewIndex === i} 
                onClick={() => setViewIndex(viewIndex === i ? null : i)}
                selectionMap={batchSelectionState.get(i)}
              />
            ))}
          </div>

          <button onClick={() => exportBatchAsZIP(batchResults)} style={{ ...btnSecondary, alignSelf: 'flex-start', marginTop: '0.5rem' }}>
            批量导出 ZIP
          </button>

          {/* Detail view */}
          {viewIndex !== null && batchResults[viewIndex]?.result && (
            <ResultDetail 
              batchResult={batchResults[viewIndex]} 
              threshold={filterThreshold}
              onThresholdChange={setFilterThreshold}
              selectionMap={currentSelectionMap}
              onSelectionChange={handleSelectionChange}
            />
          )}
        </section>
      )}
    </div>
  );
}


/** Thumbnail card for a batch result */
function ResultCard({ batchResult, isSelected, onClick, selectionMap }: {
  batchResult: BatchResult;
  isSelected: boolean;
  onClick: () => void;
  selectionMap?: Map<number, boolean>;
}) {
  const [thumbUrl, setThumbUrl] = useState<string | null>(null);
  const hasResult = batchResult.result && batchResult.result.count > 0;
  
  // Calculate selected count from selectionMap
  const totalCount = batchResult.result?.count ?? 0;
  const selectedCount = selectionMap 
    ? Array.from(selectionMap.values()).filter(v => v).length 
    : totalCount;

  // Create and cleanup blob URL
  useEffect(() => {
    if (!batchResult.file) return;
    const url = URL.createObjectURL(batchResult.file);
    setThumbUrl(url);
    return () => {
      URL.revokeObjectURL(url);
      setThumbUrl(null);
    };
  }, [batchResult.file]);

  return (
    <div
      onClick={onClick}
      style={{
        cursor: 'pointer',
        border: isSelected ? '2px solid #4ECDC4' : batchResult.error ? '2px solid #FF6B6B' : '1px solid #555',
        borderRadius: 6,
        overflow: 'hidden',
        background: '#2a2a2a',
        transition: 'border-color 0.2s',
      }}
    >
      {thumbUrl ? (
        <img src={thumbUrl} alt={batchResult.file.name} style={{ width: '100%', height: 100, objectFit: 'cover', display: 'block' }} />
      ) : (
        <div style={{ width: '100%', height: 100, background: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
          加载中...
        </div>
      )}
      <div style={{ padding: '0.35rem 0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.75rem', color: '#aaa', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
          {batchResult.file.name}
        </span>
        {hasResult && (
          <span style={{ fontSize: '0.75rem', color: '#4ECDC4', fontWeight: 600 }}>
            {selectedCount}/{totalCount} 选中
          </span>
        )}
        {batchResult.error && (
          <span style={{ fontSize: '0.75rem', color: '#FF6B6B' }}>失败</span>
        )}
        {batchResult.result && batchResult.result.count === 0 && !batchResult.error && (
          <span style={{ fontSize: '0.75rem', color: '#FFEAA7' }}>0 个</span>
        )}
      </div>
    </div>
  );
}

/** Detail view for a single batch result with mask overlay */
function ResultDetail({ 
  batchResult,
  threshold,
  onThresholdChange,
  selectionMap,
  onSelectionChange,
}: { 
  batchResult: BatchResult;
  threshold: number;
  onThresholdChange: (threshold: number) => void;
  selectionMap: Map<number, boolean>;
  onSelectionChange: (selectionMap: Map<number, boolean>) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);

  const result = batchResult.result;
  
  // Compute colored masks with original indices (safe even if result is null)
  const coloredMasks: (MaskData & { originalIndex: number })[] = result
    ? (() => {
        const colors = assignMaskColors(result.masks.length);
        return result.masks.map((m, i) => ({ ...m, color: colors[i], originalIndex: i }));
      })()
    : [];

  // Sort masks by confidence (descending) - Requirement 2.1
  const sortedMasks = sortByConfidence(coloredMasks);

  // Filter masks by threshold - Requirement 1.2
  const filteredMasks = filterByConfidence(sortedMasks, threshold);

  // Get filtered indices for batch operations
  const filteredIndices = filteredMasks.map(m => m.originalIndex);

  // Calculate filter stats - Requirement 5.1
  const stats = getFilterStats(result?.masks ?? [], threshold, selectionMap);

  // Helper to check if a mask is selected (default to true)
  const isMaskSelected = (originalIndex: number) => selectionMap.get(originalIndex) ?? true;

  // Toggle selection for a mask
  const toggleSelection = (originalIndex: number) => {
    const newMap = new Map(selectionMap);
    const current = newMap.get(originalIndex) ?? true;
    newMap.set(originalIndex, !current);
    onSelectionChange(newMap);
  };

  // Batch operations
  const handleSelectAll = () => {
    const newMap = new Map(selectionMap);
    for (const index of filteredIndices) {
      newMap.set(index, true);
    }
    onSelectionChange(newMap);
  };

  const handleSelectNone = () => {
    const newMap = new Map(selectionMap);
    for (const key of newMap.keys()) {
      newMap.set(key, false);
    }
    // Also set filtered indices to false
    for (const index of filteredIndices) {
      newMap.set(index, false);
    }
    onSelectionChange(newMap);
  };

  const handleInvertSelection = () => {
    const newMap = new Map(selectionMap);
    for (const index of filteredIndices) {
      const current = newMap.get(index) ?? true;
      newMap.set(index, !current);
    }
    onSelectionChange(newMap);
  };

  // Create stable blob URL for the file
  useEffect(() => {
    if (!batchResult.file) return;
    
    const url = URL.createObjectURL(batchResult.file);
    setImgUrl(url);
    setImgEl(null); // Reset image when file changes
    return () => {
      URL.revokeObjectURL(url);
      setImgUrl(null);
    };
  }, [batchResult.file]);

  // Load image from blob URL
  useEffect(() => {
    if (!imgUrl) return;
    
    const img = new Image();
    img.onload = () => setImgEl(img);
    img.onerror = (e) => console.error('Failed to load image from blob URL:', imgUrl, e);
    img.src = imgUrl;
  }, [imgUrl]);

  // Compute scaled dimensions - only when image is loaded
  const cW = imgEl
    ? calculateScaledSize(imgEl.naturalWidth, imgEl.naturalHeight, RESULT_CANVAS_MAX, RESULT_CANVAS_MAX).scaledWidth
    : RESULT_CANVAS_MAX;
  const cH = imgEl
    ? calculateScaledSize(imgEl.naturalWidth, imgEl.naturalHeight, RESULT_CANVAS_MAX, RESULT_CANVAS_MAX).scaledHeight
    : 300;

  // Store mask pixel data for click detection
  const maskPixelDataRef = useRef<Map<number, ImageData>>(new Map());

  // Draw result canvas with mask overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl || !result) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match scaled image
    canvas.width = cW;
    canvas.height = cH;

    // Draw the base image first
    ctx.clearRect(0, 0, cW, cH);
    ctx.drawImage(imgEl, 0, 0, cW, cH);

    // If no masks, we're done
    if (filteredMasks.length === 0) {
      maskPixelDataRef.current.clear();
      return;
    }

    // Load all mask images
    const maskImgs: HTMLImageElement[] = [];
    let loadedCount = 0;

    const drawAllMasks = () => {
      // Redraw base image to ensure clean canvas
      ctx.clearRect(0, 0, cW, cH);
      ctx.drawImage(imgEl, 0, 0, cW, cH);
      
      // Clear and rebuild mask pixel data for click detection
      maskPixelDataRef.current.clear();
      
      // Draw each mask overlay
      for (let i = 0; i < maskImgs.length; i++) {
        const mi = maskImgs[i];
        const mask = filteredMasks[i];
        if (!mi.complete || mi.naturalWidth === 0) {
          continue;
        }
        
        const isSelected = isMaskSelected(mask.originalIndex);
        
        // Create a canvas to extract mask data
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = cW;
        maskCanvas.height = cH;
        const maskCtx = maskCanvas.getContext('2d');
        if (!maskCtx) continue;
        
        // Draw mask scaled to canvas size
        maskCtx.drawImage(mi, 0, 0, cW, cH);
        
        // Get mask pixel data
        const maskData = maskCtx.getImageData(0, 0, cW, cH);
        
        // Store mask pixel data for click detection
        maskPixelDataRef.current.set(mask.originalIndex, maskData);
        
        // Create colored overlay using mask as alpha
        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = cW;
        overlayCanvas.height = cH;
        const overlayCtx = overlayCanvas.getContext('2d');
        if (!overlayCtx) continue;
        
        // Parse the color
        const color = mask.color;
        let r = 0, g = 0, b = 0;
        if (color.startsWith('#')) {
          r = parseInt(color.slice(1, 3), 16);
          g = parseInt(color.slice(3, 5), 16);
          b = parseInt(color.slice(5, 7), 16);
        } else if (color.startsWith('rgb')) {
          const match = color.match(/\d+/g);
          if (match) {
            r = parseInt(match[0]);
            g = parseInt(match[1]);
            b = parseInt(match[2]);
          }
        }
        
        // Create overlay image data
        const overlayData = overlayCtx.createImageData(cW, cH);
        for (let j = 0; j < maskData.data.length; j += 4) {
          const maskValue = maskData.data[j]; // R channel of grayscale mask
          overlayData.data[j] = r;
          overlayData.data[j + 1] = g;
          overlayData.data[j + 2] = b;
          overlayData.data[j + 3] = maskValue;
        }
        
        overlayCtx.putImageData(overlayData, 0, 0);
        
        // Draw overlay with transparency - reduced for unselected masks (Requirement 3.4)
        ctx.save();
        ctx.globalAlpha = isSelected ? 0.45 : 0.15;
        ctx.drawImage(overlayCanvas, 0, 0);
        ctx.restore();
      }

      // Draw bboxes and labels
      const sx = cW / imgEl.naturalWidth;
      const sy = cH / imgEl.naturalHeight;
      
      for (let i = 0; i < filteredMasks.length; i++) {
        const m = filteredMasks[i];
        const isSelected = isMaskSelected(m.originalIndex);
        const [bx1, by1, bx2, by2] = m.bbox;
        
        ctx.strokeStyle = isSelected ? m.color : '#666';
        ctx.lineWidth = isSelected ? 1.5 : 1;
        ctx.strokeRect(bx1 * sx, by1 * sy, (bx2 - bx1) * sx, (by2 - by1) * sy);
        
        // Score label
        const label = `${(m.score * 100).toFixed(1)}%`;
        ctx.fillStyle = isSelected ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.4)';
        const tx = bx1 * sx;
        const ty = Math.max(14, by1 * sy - 3);
        ctx.fillRect(tx, ty - 12, 42, 15);
        ctx.fillStyle = isSelected ? '#fff' : '#888';
        ctx.font = '10px sans-serif';
        ctx.fillText(label, tx + 2, ty);
      }
    };

    // Load mask images
    filteredMasks.forEach((m, i) => {
      const mi = new Image();
      mi.onload = () => {
        console.log(`Mask ${i} loaded: naturalSize=${mi.naturalWidth}x${mi.naturalHeight}, bbox=${JSON.stringify(m.bbox)}`);
        loadedCount++;
        if (loadedCount === filteredMasks.length) {
          drawAllMasks();
        }
      };
      mi.onerror = () => {
        loadedCount++;
        if (loadedCount === filteredMasks.length) {
          drawAllMasks();
        }
      };
      mi.src = `data:image/png;base64,${m.maskBase64}`;
      maskImgs[i] = mi;
    });
  }, [imgEl, filteredMasks, cW, cH, result, selectionMap]);

  // Handle canvas click to toggle mask selection (Requirement 3.3)
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(e.clientX - rect.left);
    const y = Math.floor(e.clientY - rect.top);
    
    // Check each mask's pixel data to see if click is inside
    // Check in reverse order (top masks first) to handle overlapping
    for (let i = filteredMasks.length - 1; i >= 0; i--) {
      const mask = filteredMasks[i];
      const maskData = maskPixelDataRef.current.get(mask.originalIndex);
      if (!maskData) continue;
      
      const pixelIndex = (y * cW + x) * 4;
      const maskValue = maskData.data[pixelIndex]; // R channel
      
      if (maskValue > 128) { // Threshold for mask detection
        toggleSelection(mask.originalIndex);
        return;
      }
    }
  }, [filteredMasks, cW, imgEl, toggleSelection]);

  // Early return after all hooks
  if (!result) return null;

  return (
    <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: '#1e1e1e', borderRadius: 6, border: '1px solid #444' }}>
      {/* Filter Controls - Requirement 1.1 */}
      <FilterControls
        threshold={threshold}
        onThresholdChange={onThresholdChange}
        stats={stats}
        onSelectAll={handleSelectAll}
        onSelectNone={handleSelectNone}
        onInvertSelection={handleInvertSelection}
        disabled={coloredMasks.length === 0}
      />
      
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          {imgEl ? (
            <canvas 
              ref={canvasRef} 
              width={cW} 
              height={cH} 
              style={{ border: '1px solid #555', borderRadius: 4, display: 'block', cursor: 'pointer' }} 
              onClick={handleCanvasClick}
            />
          ) : (
            <div style={{ width: RESULT_CANVAS_MAX, height: 300, border: '1px solid #555', borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
              加载中...
            </div>
          )}
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.9rem' }}>
            <span style={{ color: '#4ECDC4', fontWeight: 600 }}>{batchResult.file.name}</span>
          </p>
          <p style={{ margin: '0 0 0.25rem', fontSize: '0.85rem', color: '#ccc' }}>
            检测到 {result.count} 个对象 · 耗时 {result.processingTimeMs.toFixed(0)}ms
          </p>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', color: '#aaa' }}>
            图像尺寸: {result.imageSize[0]} × {result.imageSize[1]}
          </p>
          {/* Mask list sorted by confidence with checkboxes - Requirements 2.1, 2.2, 3.1, 3.2 */}
          {filteredMasks.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', maxHeight: 200, overflowY: 'auto' }}>
              {filteredMasks.map((m) => {
                const isSelected = isMaskSelected(m.originalIndex);
                return (
                  <div 
                    key={m.originalIndex} 
                    style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '0.4rem', 
                      fontSize: '0.8rem',
                      opacity: isSelected ? 1 : 0.5,
                      cursor: 'pointer',
                      padding: '0.15rem 0.25rem',
                      borderRadius: 3,
                      background: isSelected ? 'rgba(78, 205, 196, 0.1)' : 'transparent',
                    }}
                    onClick={() => toggleSelection(m.originalIndex)}
                  >
                    {/* Checkbox - Requirement 3.1 */}
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSelection(m.originalIndex)}
                      onClick={(e) => e.stopPropagation()}
                      style={{ cursor: 'pointer' }}
                    />
                    <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: m.color }} />
                    <span>对象 {m.originalIndex + 1}</span>
                    <span style={{ color: '#aaa' }}>{(m.score * 100).toFixed(1)}%</span>
                    <span style={{ color: '#666' }}>面积: {m.area.toLocaleString()}px</span>
                  </div>
                );
              })}
            </div>
          ) : coloredMasks.length > 0 ? (
            <p style={{ fontSize: '0.8rem', color: '#FFEAA7', margin: 0 }}>
              当前阈值下无结果，请降低阈值
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// --- Styles ---
const sectionStyle: React.CSSProperties = {
  padding: '1rem',
  border: '1px solid #444',
  borderRadius: 8,
  background: '#2a2a2a',
};

const stepTitleStyle: React.CSSProperties = {
  margin: '0 0 0.75rem',
  fontSize: '1rem',
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
};

const stepBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 24,
  height: 24,
  borderRadius: '50%',
  background: '#4ECDC4',
  color: '#000',
  fontSize: '0.8rem',
  fontWeight: 700,
  flexShrink: 0,
};

const btnPrimary: React.CSSProperties = {
  padding: '0.45rem 1rem',
  borderRadius: 4,
  border: 'none',
  background: '#4ECDC4',
  color: '#000',
  fontWeight: 600,
  cursor: 'pointer',
  fontSize: '0.9rem',
};

const btnSecondary: React.CSSProperties = {
  padding: '0.35rem 0.7rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: 'transparent',
  color: 'inherit',
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const btnDanger: React.CSSProperties = {
  padding: '0.35rem 0.7rem',
  borderRadius: 4,
  border: '1px solid #FF6B6B',
  background: 'transparent',
  color: '#FF6B6B',
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const btnDisabled: React.CSSProperties = {
  ...btnPrimary,
  background: '#555',
  cursor: 'not-allowed',
};

const btnModeActive: React.CSSProperties = {
  padding: '0.5rem 1rem',
  borderRadius: 4,
  border: '2px solid #4ECDC4',
  background: 'rgba(78, 205, 196, 0.15)',
  color: '#4ECDC4',
  fontWeight: 600,
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const btnModeInactive: React.CSSProperties = {
  padding: '0.5rem 1rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: 'transparent',
  color: '#aaa',
  fontWeight: 400,
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const textInputStyle: React.CSSProperties = {
  padding: '0.6rem 0.8rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: '#1e1e1e',
  color: '#fff',
  fontSize: '0.9rem',
  width: '100%',
  maxWidth: 400,
};
