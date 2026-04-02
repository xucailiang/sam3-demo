import { useCallback, useRef, useState, useEffect } from 'react';
import type { BoxPrompt, MaskData, BatchResult, WorkflowMode, DefectBox } from '../types';
import { validateFileExtension } from '../utils/validation';
import { assignMaskColors, calculateScaledSize } from '../utils/canvas';
import { useBatchSegmentation } from '../hooks/useBatchSegmentation';
import { useSegmentation } from '../hooks/useSegmentation';
import { useSampleApi } from '../hooks/useSampleApi';
import { exportBatchAsZIP } from '../utils/export';
import { FilterControls } from './FilterControls';
import { sortByConfidence, filterByConfidence, getFilterStats } from '../utils/filter';

// 预定义的 BBox 颜色列表
const DEFECT_BOX_COLORS = [
  '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
  '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA',
];

const generateId = () => Math.random().toString(36).substring(2, 9);

const SAMPLE_CANVAS_MAX = 420;
const RESULT_CANVAS_MAX = 420;

/**
 * SampleWorkflow - 左右分栏布局
 * 左侧：输入区（样品图 + BBox + 批量上传）
 * 右侧：结果区（详情 + 缩略图网格）
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

  // === Multi-BBox state ===
  const [defectBoxes, setDefectBoxes] = useState<DefectBox[]>([]);
  const [pendingBox, setPendingBox] = useState<BoxPrompt | null>(null);
  const [categoryInput, setCategoryInput] = useState<string>('');
  const [showCategoryDialog, setShowCategoryDialog] = useState<boolean>(false);
  const [editingBoxId, setEditingBoxId] = useState<string | null>(null);

  // === Sample API hook ===
  const {
    sampleId, featureTimeMs, isLoading: sampleApiLoading, error: sampleApiError,
    createSample, inferWithSample, deleteSample, clearSample,
  } = useSampleApi();

  const { result: sampleResult, isLoading: sampleLoading, error: sampleError, segmentWithBoxes, clearResult: clearSampleResult } = useSegmentation();

  // === Mode B: Text prompt ===
  const [textPrompt, setTextPrompt] = useState<string>('');

  // === Batch images ===
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const { results: batchResults, progress, isProcessing, batchSegmentWithText, batchSegmentWithStitch, batchSegmentWithSample, cancelBatch, clearResults } = useBatchSegmentation();

  // === View detail ===
  const [viewIndex, setViewIndex] = useState<number | null>(null);

  // === Filter and selection state ===
  const [filterThreshold, setFilterThreshold] = useState<number>(0.80);
  const [batchSelectionState, setBatchSelectionState] = useState<Map<number, Map<number, boolean>>>(new Map());
  const [inferenceConfidence, setInferenceConfidence] = useState<number>(0.25);

  const currentSelectionMap = viewIndex !== null 
    ? (batchSelectionState.get(viewIndex) ?? new Map<number, boolean>())
    : new Map<number, boolean>();

  const handleSelectionChange = useCallback((newSelectionMap: Map<number, boolean>) => {
    if (viewIndex === null) return;
    setBatchSelectionState(prev => {
      const next = new Map(prev);
      next.set(viewIndex, newSelectionMap);
      return next;
    });
  }, [viewIndex]);

  const [uploadError, setUploadError] = useState<string | null>(null);

  // --- Mode switch handler ---
  const handleModeSwitch = useCallback((newMode: WorkflowMode) => {
    if (newMode === mode) return;
    setMode(newMode);
    setSampleFile(null); setSampleImg(null); setSampleBox(null);
    setDragStart(null); setDragCurrent(null);
    clearSampleResult();
    setDefectBoxes([]); clearSample();
    setPendingBox(null); setCategoryInput(''); setShowCategoryDialog(false); setEditingBoxId(null);
    setTextPrompt('');
    setBatchFiles([]); clearResults(); setViewIndex(null); setUploadError(null);
    setFilterThreshold(0.80); setBatchSelectionState(new Map());
  }, [mode, clearSampleResult, clearResults, clearSample]);

  // --- Sample image upload ---
  const handleSampleUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!validateFileExtension(file.name)) {
      setUploadError('不支持的文件格式');
      return;
    }
    setUploadError(null);
    setSampleFile(file); setSampleBox(null);
    clearSampleResult(); clearResults(); setBatchFiles([]); setViewIndex(null);
    setDefectBoxes([]); clearSample();
    setPendingBox(null); setCategoryInput(''); setShowCategoryDialog(false); setEditingBoxId(null);

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setSampleImg(img);
    img.src = url;
  }, [clearSampleResult, clearResults, clearSample]);

  // --- Sample canvas dimensions ---
  const sampleScaled = sampleImg
    ? calculateScaledSize(sampleImg.naturalWidth, sampleImg.naturalHeight, SAMPLE_CANVAS_MAX, SAMPLE_CANVAS_MAX)
    : { scaledWidth: SAMPLE_CANVAS_MAX, scaledHeight: 280 };
  const sW = sampleScaled.scaledWidth;
  const sH = sampleScaled.scaledHeight;
  const sScaleX = sampleImg ? sampleImg.naturalWidth / sW : 1;
  const sScaleY = sampleImg ? sampleImg.naturalHeight / sH : 1;

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
          off.width = sW; off.height = sH;
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

    // Draw all defect boxes
    for (const box of defectBoxes) {
      const x = box.x1 / sScaleX, y = box.y1 / sScaleY;
      const w = (box.x2 - box.x1) / sScaleX, h = (box.y2 - box.y1) / sScaleY;
      ctx.strokeStyle = box.color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      const labelText = box.category;
      ctx.font = '11px sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = box.color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, y - 18, Math.max(textWidth + 8, 40), 18);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#000';
      ctx.fillText(labelText, x + 4, y - 5);
    }

    // Draw single box (legacy)
    if (sampleBox && defectBoxes.length === 0) {
      const x = sampleBox.x1 / sScaleX, y = sampleBox.y1 / sScaleY;
      const w = (sampleBox.x2 - sampleBox.x1) / sScaleX, h = (sampleBox.y2 - sampleBox.y1) / sScaleY;
      ctx.strokeStyle = '#FF6B6B';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(255,107,107,0.85)';
      ctx.fillRect(x, y - 18, 60, 18);
      ctx.fillStyle = '#fff';
      ctx.font = '11px sans-serif';
      ctx.fillText('标记区域', x + 4, y - 5);
    }

    // Draw pending box
    if (pendingBox && showCategoryDialog) {
      const x = pendingBox.x1 / sScaleX, y = pendingBox.y1 / sScaleY;
      const w = (pendingBox.x2 - pendingBox.x1) / sScaleX, h = (pendingBox.y2 - pendingBox.y1) / sScaleY;
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,0,0.85)';
      ctx.fillRect(x, y - 18, 70, 18);
      ctx.fillStyle = '#000';
      ctx.font = '11px sans-serif';
      ctx.fillText('待确认...', x + 4, y - 5);
    }

    // Draw in-progress drag
    if (dragStart && dragCurrent) {
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(dragStart.x, dragStart.y, dragCurrent.x - dragStart.x, dragCurrent.y - dragStart.y);
      ctx.setLineDash([]);
    }
  }, [sampleImg, sW, sH, sampleBox, sScaleX, sScaleY, dragStart, dragCurrent, sampleMasks, defectBoxes, pendingBox, showCategoryDialog]);

  useEffect(() => { drawSampleCanvas(); }, [drawSampleCanvas]);

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
      setDragStart(null); setDragCurrent(null);
      return;
    }
    let x1 = dragStart.x * sScaleX, y1 = dragStart.y * sScaleY;
    let x2 = dragCurrent.x * sScaleX, y2 = dragCurrent.y * sScaleY;
    if (x1 > x2) [x1, x2] = [x2, x1];
    if (y1 > y2) [y1, y2] = [y2, y1];
    x1 = Math.max(0, x1); y1 = Math.max(0, y1);
    x2 = Math.min(sampleImg.naturalWidth, x2); y2 = Math.min(sampleImg.naturalHeight, y2);
    if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
      setPendingBox({ x1, y1, x2, y2 });
      setCategoryInput('');
      setShowCategoryDialog(true);
      setSampleBox({ x1, y1, x2, y2 });
    }
    setDragStart(null); setDragCurrent(null);
  }, [dragStart, dragCurrent, sampleImg, sScaleX, sScaleY]);


  // --- Multi-BBox handlers ---
  const handleConfirmBox = useCallback(() => {
    if (!pendingBox || !categoryInput.trim()) return;
    const newBox: DefectBox = {
      id: generateId(),
      x1: pendingBox.x1, y1: pendingBox.y1, x2: pendingBox.x2, y2: pendingBox.y2,
      category: categoryInput.trim(),
      color: DEFECT_BOX_COLORS[defectBoxes.length % DEFECT_BOX_COLORS.length],
    };
    setDefectBoxes(prev => [...prev, newBox]);
    setPendingBox(null); setCategoryInput(''); setShowCategoryDialog(false);
    clearSample();
  }, [pendingBox, categoryInput, defectBoxes.length, clearSample]);

  const handleCancelBox = useCallback(() => {
    setPendingBox(null); setCategoryInput(''); setShowCategoryDialog(false);
    if (defectBoxes.length === 0) setSampleBox(null);
  }, [defectBoxes.length]);

  const handleDeleteBox = useCallback((boxId: string) => {
    setDefectBoxes(prev => {
      const newBoxes = prev.filter(b => b.id !== boxId);
      return newBoxes.map((b, i) => ({ ...b, color: DEFECT_BOX_COLORS[i % DEFECT_BOX_COLORS.length] }));
    });
    clearSample();
    if (defectBoxes.length <= 1) setSampleBox(null);
  }, [defectBoxes.length, clearSample]);

  const handleStartEditBox = useCallback((boxId: string) => {
    const box = defectBoxes.find(b => b.id === boxId);
    if (box) { setEditingBoxId(boxId); setCategoryInput(box.category); }
  }, [defectBoxes]);

  const handleConfirmEditBox = useCallback(() => {
    if (!editingBoxId || !categoryInput.trim()) return;
    setDefectBoxes(prev => prev.map(b => b.id === editingBoxId ? { ...b, category: categoryInput.trim() } : b));
    setEditingBoxId(null); setCategoryInput('');
    clearSample();
  }, [editingBoxId, categoryInput, clearSample]);

  const handleCancelEditBox = useCallback(() => { setEditingBoxId(null); setCategoryInput(''); }, []);

  const handleClearAllBoxes = useCallback(() => {
    setDefectBoxes([]); setSampleBox(null); clearSample(); clearSampleResult();
  }, [clearSampleResult, clearSample]);

  const handlePreviewSample = useCallback(async () => {
    if (!sampleFile || !sampleBox) return;
    await segmentWithBoxes(sampleFile, [sampleBox]);
  }, [sampleFile, sampleBox, segmentWithBoxes]);

  const handleCreateSample = useCallback(async () => {
    if (!sampleFile || defectBoxes.length === 0) return;
    await createSample(sampleFile, defectBoxes);
  }, [sampleFile, defectBoxes, createSample]);

  const handleDeleteSample = useCallback(async () => {
    if (!sampleId) return;
    await deleteSample(sampleId);
  }, [sampleId, deleteSample]);

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
    setFilterThreshold(0.80);
    setBatchSelectionState(new Map());
    
    if (mode === 'stitch') {
      if (sampleId && defectBoxes.length > 0 && batchFiles.length > 0) {
        await batchSegmentWithSample(batchFiles, (files) => inferWithSample(sampleId, files, inferenceConfidence));
      } else if (sampleFile && sampleBox && batchFiles.length > 0) {
        await batchSegmentWithStitch(batchFiles, sampleFile, sampleBox, inferenceConfidence);
      }
    } else {
      if (!textPrompt.trim() || batchFiles.length === 0) return;
      await batchSegmentWithText(batchFiles, textPrompt.trim());
    }
  }, [mode, sampleId, defectBoxes.length, sampleFile, sampleBox, batchFiles, textPrompt, batchSegmentWithStitch, batchSegmentWithText, batchSegmentWithSample, inferenceConfidence, inferWithSample]);

  const canBatchModeA = (
    (sampleId && defectBoxes.length > 0 && batchFiles.length > 0) ||
    (sampleFile !== null && sampleBox !== null && batchFiles.length > 0)
  ) && !isProcessing && !sampleApiLoading;
  const canBatchModeB = textPrompt.trim().length > 0 && batchFiles.length > 0 && !isProcessing;
  const canBatch = mode === 'stitch' ? canBatchModeA : canBatchModeB;

  // Auto-select first result when batch completes
  useEffect(() => {
    if (batchResults.length > 0 && viewIndex === null) {
      const firstSuccess = batchResults.findIndex(r => r.result && r.result.count > 0);
      if (firstSuccess >= 0) setViewIndex(firstSuccess);
    }
  }, [batchResults, viewIndex]);


  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Mode Switcher */}
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
        <button onClick={() => handleModeSwitch('stitch')} style={mode === 'stitch' ? btnModeActive : btnModeInactive}>
          模式 A：图像拼接 + BBox
        </button>
        <button onClick={() => handleModeSwitch('text')} style={mode === 'text' ? btnModeActive : btnModeInactive}>
          模式 B：纯文本
        </button>
      </div>

      {/* Main two-column layout */}
      <div style={layoutContainer}>
        {/* === LEFT COLUMN: Input Area === */}
        <div style={leftColumn}>
          {/* Section 1: Sample Image / Text Input */}
          <div style={sectionStyle}>
            <h3 style={sectionTitle}>
              <span style={stepBadge}>1</span>
              {mode === 'stitch' ? '上传样品图并标记目标' : '输入文本描述'}
            </h3>

            {mode === 'stitch' ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                  <input ref={sampleInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleSampleUpload} style={{ display: 'none' }} />
                  <button onClick={() => sampleInputRef.current?.click()} style={btnPrimary} disabled={sampleLoading}>选择样品图</button>
                  {sampleFile && <span style={{ fontSize: '0.8rem', color: '#aaa' }}>{sampleFile.name}</span>}
                </div>

                {sampleImg ? (
                  <>
                    <p style={{ fontSize: '0.8rem', color: '#888', margin: '0 0 0.4rem' }}>拖拽绘制边界框，框选目标区域</p>
                    <canvas
                      ref={sampleCanvasRef}
                      width={sW} height={sH}
                      style={{ border: '1px solid #555', cursor: 'crosshair', display: 'block', borderRadius: 4, maxWidth: '100%' }}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={() => { if (dragStart) { setDragStart(null); setDragCurrent(null); } }}
                    />
                  </>
                ) : (
                  <div style={placeholderBox}>请上传样品图</div>
                )}

                {/* Category Input Dialog */}
                {showCategoryDialog && pendingBox && (
                  <div style={categoryDialogStyle}>
                    <p style={{ margin: '0 0 0.4rem', fontSize: '0.85rem' }}>输入缺陷类别：</p>
                    <div style={{ display: 'flex', gap: '0.4rem' }}>
                      <input
                        type="text" value={categoryInput}
                        onChange={(e) => setCategoryInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && categoryInput.trim()) handleConfirmBox(); if (e.key === 'Escape') handleCancelBox(); }}
                        placeholder="例如：划痕、气泡..."
                        style={{ ...textInputStyle, flex: 1 }} autoFocus
                      />
                      <button onClick={handleConfirmBox} disabled={!categoryInput.trim()} style={categoryInput.trim() ? btnPrimary : btnDisabled}>确认</button>
                      <button onClick={handleCancelBox} style={btnSecondary}>取消</button>
                    </div>
                  </div>
                )}

                {/* Defect Box List */}
                {defectBoxes.length > 0 && (
                  <div style={boxListContainer}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                      <span style={{ fontSize: '0.8rem', color: '#ccc' }}>已标记 {defectBoxes.length} 个区域</span>
                      <button onClick={handleClearAllBoxes} style={btnDangerSmall}>清除全部</button>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', maxHeight: 120, overflowY: 'auto' }}>
                      {defectBoxes.map((box) => (
                        <div key={box.id} style={{ ...boxItemStyle, borderLeftColor: box.color }}>
                          {editingBoxId === box.id ? (
                            <>
                              <input type="text" value={categoryInput} onChange={(e) => setCategoryInput(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter' && categoryInput.trim()) handleConfirmEditBox(); if (e.key === 'Escape') handleCancelEditBox(); }}
                                style={{ ...textInputStyle, flex: 1, padding: '0.2rem 0.4rem', fontSize: '0.75rem' }} autoFocus />
                              <button onClick={handleConfirmEditBox} disabled={!categoryInput.trim()} style={btnTiny}>保存</button>
                              <button onClick={handleCancelEditBox} style={btnTinySecondary}>取消</button>
                            </>
                          ) : (
                            <>
                              <span style={{ width: 10, height: 10, borderRadius: 2, background: box.color, flexShrink: 0 }} />
                              <span style={{ flex: 1, fontSize: '0.8rem' }}>{box.category}</span>
                              <button onClick={() => handleStartEditBox(box.id)} style={btnTinySecondary}>编辑</button>
                              <button onClick={() => handleDeleteBox(box.id)} style={btnTinyDanger}>删除</button>
                            </>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Create Sample Button */}
                    <div style={{ marginTop: '0.5rem', paddingTop: '0.4rem', borderTop: '1px solid #444' }}>
                      {!sampleId ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                          <button onClick={handleCreateSample} disabled={sampleApiLoading || defectBoxes.length === 0}
                            style={sampleApiLoading || defectBoxes.length === 0 ? btnDisabled : btnPrimary}>
                            {sampleApiLoading ? '创建中...' : '创建样本'}
                          </button>
                          <span style={{ fontSize: '0.75rem', color: '#888' }}>缓存特征以加速推理</span>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '0.8rem', color: '#4ECDC4' }}>✓ 样本已创建</span>
                          {featureTimeMs && <span style={{ fontSize: '0.75rem', color: '#888' }}>{featureTimeMs.toFixed(0)}ms</span>}
                          <button onClick={handleDeleteSample} disabled={sampleApiLoading} style={btnDangerSmall}>删除</button>
                        </div>
                      )}
                      {sampleApiError && <p style={{ color: '#FF6B6B', fontSize: '0.75rem', margin: '0.25rem 0 0' }}>{sampleApiError}</p>}
                    </div>
                  </div>
                )}

                {/* Legacy single box */}
                {sampleBox && defectBoxes.length === 0 && !showCategoryDialog && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.8rem', color: '#aaa' }}>
                      [{Math.round(sampleBox.x1)}, {Math.round(sampleBox.y1)}, {Math.round(sampleBox.x2)}, {Math.round(sampleBox.y2)}]
                    </span>
                    <button onClick={() => { setSampleBox(null); clearSampleResult(); }} style={btnDangerSmall}>重新标记</button>
                    <button onClick={handlePreviewSample} disabled={sampleLoading} style={btnSecondary}>
                      {sampleLoading ? '预览中...' : '预览效果'}
                    </button>
                  </div>
                )}

                {sampleError && <p style={{ color: '#FF6B6B', fontSize: '0.8rem', margin: '0.25rem 0 0' }}>{sampleError}</p>}
                {sampleResult && sampleResult.masks.length > 0 && (
                  <div style={{ fontSize: '0.8rem', color: '#4ECDC4', marginTop: '0.25rem' }}>
                    ✓ 检测到 {sampleResult.count} 个对象
                  </div>
                )}
              </>
            ) : (
              /* Mode B: Text Input */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <input type="text" value={textPrompt} onChange={(e) => setTextPrompt(e.target.value)}
                  placeholder="例如：cat, dog, person..." style={textInputStyle} />
                {!textPrompt.trim() && <p style={{ fontSize: '0.75rem', color: '#FFEAA7', margin: 0 }}>请输入文本描述</p>}
              </div>
            )}
          </div>


          {/* Section 2: Batch Upload & Process */}
          {(mode === 'text' || (mode === 'stitch' && sampleBox)) && (
            <div style={sectionStyle}>
              <h3 style={sectionTitle}>
                <span style={stepBadge}>2</span>
                批量分割
              </h3>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
                <input ref={batchInputRef} type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={handleBatchUpload} style={{ display: 'none' }} />
                <button onClick={() => batchInputRef.current?.click()} disabled={isProcessing} style={btnPrimary}>选择图片</button>
                {batchFiles.length > 0 && <span style={{ fontSize: '0.8rem', color: '#aaa' }}>已选 {batchFiles.length} 张</span>}
                {uploadError && <span style={{ fontSize: '0.8rem', color: '#FF6B6B' }}>{uploadError}</span>}
              </div>

              {batchFiles.length > 0 && (
                <div style={{ fontSize: '0.75rem', color: '#666', maxHeight: 50, overflowY: 'auto', marginBottom: '0.4rem' }}>
                  {batchFiles.map((f, i) => <div key={i}>{f.name}</div>)}
                </div>
              )}

              {/* Inference confidence slider */}
              {mode === 'stitch' && (
                <div style={{ marginBottom: '0.5rem', padding: '0.4rem', background: '#1a1a1a', borderRadius: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <label style={{ fontSize: '0.8rem', color: '#aaa', whiteSpace: 'nowrap' }}>推理置信度</label>
                    <input type="range" min={0} max={100} step={1} value={inferenceConfidence * 100}
                      onChange={(e) => setInferenceConfidence(Number(e.target.value) / 100)}
                      disabled={isProcessing} style={{ flex: 1, minWidth: 80 }} />
                    <span style={{ fontSize: '0.8rem', color: '#4ECDC4', fontWeight: 600, minWidth: 40 }}>{(inferenceConfidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button onClick={handleBatchSegment} disabled={!canBatch} style={canBatch ? btnPrimary : btnDisabled}>
                  {isProcessing ? '处理中...' : '开始分割'}
                </button>
                {isProcessing && <button onClick={cancelBatch} style={btnDangerSmall}>取消</button>}
              </div>

              {/* Progress */}
              {progress.status !== 'idle' && (
                <div style={{ marginTop: '0.4rem' }}>
                  <div style={{ height: 5, borderRadius: 3, background: '#333', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${progress.total > 0 ? Math.round((progress.completed / progress.total) * 100) : 0}%`,
                      background: progress.status === 'error' ? '#FF6B6B' : '#4ECDC4',
                      transition: 'width 0.3s',
                    }} />
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#888' }}>
                    {progress.status === 'processing' && `${progress.completed}/${progress.total}`}
                    {progress.status === 'completed' && '完成'}
                    {progress.status === 'error' && '出错'}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* === RIGHT COLUMN: Results Area === */}
        <div style={rightColumn}>
          {batchResults.length > 0 ? (
            <>
              {/* Result Detail */}
              {viewIndex !== null && batchResults[viewIndex]?.result && (
                <ResultDetail
                  batchResult={batchResults[viewIndex]}
                  threshold={filterThreshold}
                  onThresholdChange={setFilterThreshold}
                  selectionMap={currentSelectionMap}
                  onSelectionChange={handleSelectionChange}
                />
              )}

              {/* Thumbnail Grid */}
              <div style={sectionStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h3 style={{ ...sectionTitle, margin: 0 }}>
                    结果 ({batchResults.filter((r) => r.result && r.result.count > 0).length}/{batchResults.length})
                  </h3>
                  <button onClick={() => exportBatchAsZIP(batchResults)} style={btnSecondary}>导出 ZIP</button>
                </div>
                <div style={thumbnailGrid}>
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
              </div>
            </>
          ) : (
            <div style={{ ...sectionStyle, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 300, color: '#666' }}>
              <p>分割结果将显示在这里</p>
            </div>
          )}
        </div>
      </div>
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
  const totalCount = batchResult.result?.count ?? 0;
  const selectedCount = selectionMap ? Array.from(selectionMap.values()).filter(v => v).length : totalCount;

  useEffect(() => {
    if (!batchResult.file) return;
    const url = URL.createObjectURL(batchResult.file);
    setThumbUrl(url);
    return () => { URL.revokeObjectURL(url); setThumbUrl(null); };
  }, [batchResult.file]);

  return (
    <div onClick={onClick} style={{
      cursor: 'pointer',
      border: isSelected ? '2px solid #4ECDC4' : batchResult.error ? '2px solid #FF6B6B' : '1px solid #444',
      borderRadius: 4, overflow: 'hidden', background: '#1e1e1e',
    }} title={batchResult.error ?? batchResult.file.name}>
      {thumbUrl ? (
        <img src={thumbUrl} alt={batchResult.file.name} style={{ width: '100%', height: 60, objectFit: 'cover', display: 'block' }} />
      ) : (
        <div style={{ width: '100%', height: 60, background: '#333' }} />
      )}
      <div style={{ padding: '0.2rem 0.3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.7rem' }}>
        <span style={{ color: '#888', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '60%' }}>
          {batchResult.file.name}
        </span>
        {hasResult && <span style={{ color: '#4ECDC4', fontWeight: 600 }}>{selectedCount}/{totalCount}</span>}
        {batchResult.error && <span style={{ color: '#FF6B6B' }}>✕</span>}
        {batchResult.result && batchResult.result.count === 0 && !batchResult.error && <span style={{ color: '#FFEAA7' }}>0</span>}
      </div>
    </div>
  );
}

/** Detail view for a single batch result */
function ResultDetail({ batchResult, threshold, onThresholdChange, selectionMap, onSelectionChange }: {
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
  
  const coloredMasks: (MaskData & { originalIndex: number })[] = result
    ? (() => {
        const colors = assignMaskColors(result.masks.length);
        return result.masks.map((m, i) => ({ ...m, color: colors[i], originalIndex: i }));
      })()
    : [];

  const sortedMasks = sortByConfidence(coloredMasks);
  const filteredMasks = filterByConfidence(sortedMasks, threshold);
  const filteredIndices = filteredMasks.map(m => m.originalIndex);
  const stats = getFilterStats(result?.masks ?? [], threshold, selectionMap);

  const isMaskSelected = (originalIndex: number) => selectionMap.get(originalIndex) ?? true;

  const toggleSelection = (originalIndex: number) => {
    const newMap = new Map(selectionMap);
    newMap.set(originalIndex, !(newMap.get(originalIndex) ?? true));
    onSelectionChange(newMap);
  };

  const handleSelectAll = () => {
    const newMap = new Map(selectionMap);
    for (const index of filteredIndices) newMap.set(index, true);
    onSelectionChange(newMap);
  };

  const handleSelectNone = () => {
    const newMap = new Map(selectionMap);
    for (const key of newMap.keys()) newMap.set(key, false);
    for (const index of filteredIndices) newMap.set(index, false);
    onSelectionChange(newMap);
  };

  const handleInvertSelection = () => {
    const newMap = new Map(selectionMap);
    for (const index of filteredIndices) newMap.set(index, !(newMap.get(index) ?? true));
    onSelectionChange(newMap);
  };

  useEffect(() => {
    if (!batchResult.file) return;
    const url = URL.createObjectURL(batchResult.file);
    setImgUrl(url);
    setImgEl(null);
    return () => { URL.revokeObjectURL(url); setImgUrl(null); };
  }, [batchResult.file]);

  useEffect(() => {
    if (!imgUrl) return;
    const img = new Image();
    img.onload = () => setImgEl(img);
    img.src = imgUrl;
  }, [imgUrl]);

  const cW = imgEl ? calculateScaledSize(imgEl.naturalWidth, imgEl.naturalHeight, RESULT_CANVAS_MAX, RESULT_CANVAS_MAX).scaledWidth : RESULT_CANVAS_MAX;
  const cH = imgEl ? calculateScaledSize(imgEl.naturalWidth, imgEl.naturalHeight, RESULT_CANVAS_MAX, RESULT_CANVAS_MAX).scaledHeight : 280;

  const maskPixelDataRef = useRef<Map<number, ImageData>>(new Map());


  // Draw result canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl || !result) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = cW;
    canvas.height = cH;
    ctx.clearRect(0, 0, cW, cH);
    ctx.drawImage(imgEl, 0, 0, cW, cH);

    if (filteredMasks.length === 0) {
      maskPixelDataRef.current.clear();
      return;
    }

    const maskImgs: HTMLImageElement[] = [];
    let loadedCount = 0;

    const drawAllMasks = () => {
      ctx.clearRect(0, 0, cW, cH);
      ctx.drawImage(imgEl, 0, 0, cW, cH);
      maskPixelDataRef.current.clear();

      for (let i = 0; i < maskImgs.length; i++) {
        const mi = maskImgs[i];
        const mask = filteredMasks[i];
        if (!mi.complete || mi.naturalWidth === 0) continue;

        const isSelected = isMaskSelected(mask.originalIndex);
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = cW; maskCanvas.height = cH;
        const maskCtx = maskCanvas.getContext('2d');
        if (!maskCtx) continue;

        maskCtx.drawImage(mi, 0, 0, cW, cH);
        const maskData = maskCtx.getImageData(0, 0, cW, cH);
        maskPixelDataRef.current.set(mask.originalIndex, maskData);

        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = cW; overlayCanvas.height = cH;
        const overlayCtx = overlayCanvas.getContext('2d');
        if (!overlayCtx) continue;

        const color = mask.color;
        let r = 0, g = 0, b = 0;
        if (color.startsWith('#')) {
          r = parseInt(color.slice(1, 3), 16);
          g = parseInt(color.slice(3, 5), 16);
          b = parseInt(color.slice(5, 7), 16);
        }

        const overlayData = overlayCtx.createImageData(cW, cH);
        for (let j = 0; j < maskData.data.length; j += 4) {
          const maskValue = maskData.data[j];
          overlayData.data[j] = r;
          overlayData.data[j + 1] = g;
          overlayData.data[j + 2] = b;
          overlayData.data[j + 3] = maskValue;
        }

        overlayCtx.putImageData(overlayData, 0, 0);
        ctx.save();
        ctx.globalAlpha = isSelected ? 0.45 : 0.15;
        ctx.drawImage(overlayCanvas, 0, 0);
        ctx.restore();
      }

      // Draw bboxes
      const sx = cW / imgEl.naturalWidth;
      const sy = cH / imgEl.naturalHeight;

      for (const m of filteredMasks) {
        const isSelected = isMaskSelected(m.originalIndex);
        const [bx1, by1, bx2, by2] = m.bbox;
        ctx.strokeStyle = isSelected ? m.color : '#666';
        ctx.lineWidth = isSelected ? 1.5 : 1;
        ctx.strokeRect(bx1 * sx, by1 * sy, (bx2 - bx1) * sx, (by2 - by1) * sy);

        const label = `${(m.score * 100).toFixed(0)}%`;
        ctx.fillStyle = isSelected ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.4)';
        const tx = bx1 * sx;
        const ty = Math.max(12, by1 * sy - 2);
        ctx.fillRect(tx, ty - 10, 32, 13);
        ctx.fillStyle = isSelected ? '#fff' : '#888';
        ctx.font = '9px sans-serif';
        ctx.fillText(label, tx + 2, ty);
      }
    };

    filteredMasks.forEach((m, i) => {
      const mi = new Image();
      mi.onload = () => { loadedCount++; if (loadedCount === filteredMasks.length) drawAllMasks(); };
      mi.onerror = () => { loadedCount++; if (loadedCount === filteredMasks.length) drawAllMasks(); };
      mi.src = `data:image/png;base64,${m.maskBase64}`;
      maskImgs[i] = mi;
    });
  }, [imgEl, filteredMasks, cW, cH, result, selectionMap]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(e.clientX - rect.left);
    const y = Math.floor(e.clientY - rect.top);

    for (let i = filteredMasks.length - 1; i >= 0; i--) {
      const mask = filteredMasks[i];
      const maskData = maskPixelDataRef.current.get(mask.originalIndex);
      if (!maskData) continue;
      const pixelIndex = (y * cW + x) * 4;
      if (maskData.data[pixelIndex] > 128) {
        toggleSelection(mask.originalIndex);
        return;
      }
    }
  }, [filteredMasks, cW, imgEl, toggleSelection]);

  if (!result) return null;


  return (
    <div style={sectionStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h3 style={{ ...sectionTitle, margin: 0 }}>
          <span style={{ color: '#4ECDC4' }}>{batchResult.file.name}</span>
        </h3>
        <span style={{ fontSize: '0.8rem', color: '#888' }}>
          {result.count} 个对象 · {result.processingTimeMs.toFixed(0)}ms
        </span>
      </div>

      {/* Filter Controls */}
      <FilterControls
        threshold={threshold}
        onThresholdChange={onThresholdChange}
        stats={stats}
        onSelectAll={handleSelectAll}
        onSelectNone={handleSelectNone}
        onInvertSelection={handleInvertSelection}
        disabled={coloredMasks.length === 0}
      />

      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
        {/* Canvas */}
        <div style={{ flexShrink: 0 }}>
          {imgEl ? (
            <canvas ref={canvasRef} width={cW} height={cH}
              style={{ border: '1px solid #444', borderRadius: 4, display: 'block', cursor: 'pointer' }}
              onClick={handleCanvasClick} />
          ) : (
            <div style={{ width: RESULT_CANVAS_MAX, height: 280, border: '1px solid #444', borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
              加载中...
            </div>
          )}
        </div>

        {/* Mask list */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Category stats */}
          {(() => {
            const categoryGroups = new Map<string, typeof filteredMasks>();
            for (const m of filteredMasks) {
              const cat = m.category ?? '未分类';
              if (!categoryGroups.has(cat)) categoryGroups.set(cat, []);
              categoryGroups.get(cat)!.push(m);
            }
            if (categoryGroups.size > 1 || (categoryGroups.size === 1 && !categoryGroups.has('未分类'))) {
              return (
                <div style={{ marginBottom: '0.4rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                  {Array.from(categoryGroups.entries()).map(([cat, masks]) => (
                    <span key={cat} style={{ fontSize: '0.7rem', padding: '0.1rem 0.3rem', background: 'rgba(78,205,196,0.15)', borderRadius: 3, color: '#4ECDC4' }}>
                      {cat}: {masks.length}
                    </span>
                  ))}
                </div>
              );
            }
            return null;
          })()}

          {filteredMasks.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', maxHeight: 320, overflowY: 'auto' }}>
              {filteredMasks.map((m) => {
                const isSelected = isMaskSelected(m.originalIndex);
                return (
                  <div key={m.originalIndex} onClick={() => toggleSelection(m.originalIndex)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.75rem',
                      opacity: isSelected ? 1 : 0.5, cursor: 'pointer', padding: '0.15rem 0.25rem',
                      borderRadius: 3, background: isSelected ? 'rgba(78,205,196,0.1)' : 'transparent',
                    }}>
                    <input type="checkbox" checked={isSelected} onChange={() => toggleSelection(m.originalIndex)}
                      onClick={(e) => e.stopPropagation()} style={{ cursor: 'pointer' }} />
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: m.color }} />
                    <span>#{m.originalIndex + 1}</span>
                    <span style={{ color: '#aaa' }}>{(m.score * 100).toFixed(0)}%</span>
                    {m.category && <span style={{ color: '#4ECDC4' }}>{m.category}</span>}
                  </div>
                );
              })}
            </div>
          ) : coloredMasks.length > 0 ? (
            <p style={{ fontSize: '0.75rem', color: '#FFEAA7', margin: 0 }}>当前阈值下无结果</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}


// === Styles ===
const layoutContainer: React.CSSProperties = {
  display: 'flex',
  gap: '1rem',
  alignItems: 'flex-start',
};

const leftColumn: React.CSSProperties = {
  width: 480,
  flexShrink: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '0.75rem',
};

const rightColumn: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '0.75rem',
};

const sectionStyle: React.CSSProperties = {
  padding: '0.75rem',
  border: '1px solid #444',
  borderRadius: 6,
  background: '#2a2a2a',
};

const sectionTitle: React.CSSProperties = {
  margin: '0 0 0.5rem',
  fontSize: '0.9rem',
  display: 'flex',
  alignItems: 'center',
  gap: '0.4rem',
};

const stepBadge: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 20,
  height: 20,
  borderRadius: '50%',
  background: '#4ECDC4',
  color: '#000',
  fontSize: '0.75rem',
  fontWeight: 700,
  flexShrink: 0,
};

const placeholderBox: React.CSSProperties = {
  height: 200,
  border: '2px dashed #444',
  borderRadius: 4,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#666',
  fontSize: '0.9rem',
};

const thumbnailGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
  gap: '0.5rem',
  maxHeight: 240,
  overflowY: 'auto',
};

const boxListContainer: React.CSSProperties = {
  marginTop: '0.5rem',
  padding: '0.5rem',
  background: '#1e1e1e',
  borderRadius: 4,
  border: '1px solid #444',
};

const boxItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.4rem',
  padding: '0.25rem 0.4rem',
  background: '#252525',
  borderRadius: 3,
  borderLeft: '3px solid #4ECDC4',
};

const categoryDialogStyle: React.CSSProperties = {
  marginTop: '0.5rem',
  padding: '0.5rem',
  background: '#1e1e1e',
  borderRadius: 4,
  border: '2px solid #4ECDC4',
};

const textInputStyle: React.CSSProperties = {
  padding: '0.4rem 0.6rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: '#1e1e1e',
  color: '#fff',
  fontSize: '0.85rem',
};

const btnPrimary: React.CSSProperties = {
  padding: '0.35rem 0.8rem',
  borderRadius: 4,
  border: 'none',
  background: '#4ECDC4',
  color: '#000',
  fontWeight: 600,
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const btnSecondary: React.CSSProperties = {
  padding: '0.3rem 0.6rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: 'transparent',
  color: 'inherit',
  cursor: 'pointer',
  fontSize: '0.8rem',
};

const btnDisabled: React.CSSProperties = {
  ...btnPrimary,
  background: '#555',
  cursor: 'not-allowed',
};

const btnDangerSmall: React.CSSProperties = {
  padding: '0.2rem 0.5rem',
  borderRadius: 3,
  border: '1px solid #FF6B6B',
  background: 'transparent',
  color: '#FF6B6B',
  cursor: 'pointer',
  fontSize: '0.75rem',
};

const btnTiny: React.CSSProperties = {
  padding: '0.15rem 0.4rem',
  borderRadius: 3,
  border: 'none',
  background: '#4ECDC4',
  color: '#000',
  cursor: 'pointer',
  fontSize: '0.7rem',
  fontWeight: 600,
};

const btnTinySecondary: React.CSSProperties = {
  padding: '0.15rem 0.4rem',
  borderRadius: 3,
  border: '1px solid #555',
  background: 'transparent',
  color: '#aaa',
  cursor: 'pointer',
  fontSize: '0.7rem',
};

const btnTinyDanger: React.CSSProperties = {
  padding: '0.15rem 0.4rem',
  borderRadius: 3,
  border: '1px solid #FF6B6B',
  background: 'transparent',
  color: '#FF6B6B',
  cursor: 'pointer',
  fontSize: '0.7rem',
};

const btnModeActive: React.CSSProperties = {
  padding: '0.4rem 0.8rem',
  borderRadius: 4,
  border: '2px solid #4ECDC4',
  background: 'rgba(78, 205, 196, 0.15)',
  color: '#4ECDC4',
  fontWeight: 600,
  cursor: 'pointer',
  fontSize: '0.85rem',
};

const btnModeInactive: React.CSSProperties = {
  padding: '0.4rem 0.8rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: 'transparent',
  color: '#888',
  cursor: 'pointer',
  fontSize: '0.85rem',
};
