import { useRef, useEffect, useCallback, useState } from 'react';
import type { InteractionMode, PointPrompt, BoxPrompt, MaskData } from '../types';
import { calculateScaledSize, canvasToImage as canvasToImageUtil, imageToCanvas as imageToCanvasUtil, normalizeBoundingBox } from '../utils/canvas';

export interface ImageCanvasProps {
  image: HTMLImageElement | null;
  masks: MaskData[];
  points: PointPrompt[];
  boxes: BoxPrompt[];
  mode: InteractionMode;
  maskOpacity: number;
  showMasks: boolean;
  hoveredMaskIndex?: number | null;
  onPointAdd: (point: PointPrompt) => void;
  onBoxComplete: (box: BoxPrompt) => void;
  onMaskHover: (maskIndex: number | null) => void;
}

const CANVAS_MAX_WIDTH = 800;
const CANVAS_MAX_HEIGHT = 600;

/** Foreground point marker color */
const FG_COLOR = '#00FF00';
/** Background point marker color */
const BG_COLOR = '#FF0000';
/** Point marker radius */
const POINT_RADIUS = 5;
/** Box drawing stroke color */
const BOX_STROKE = '#FFFF00';

/**
 * ImageCanvas renders the uploaded image, overlays segmentation masks,
 * handles click events for point prompts, and supports bounding box drawing.
 *
 * Requirements: 1.2, 3.1, 3.2, 3.3, 4.1
 */
export function ImageCanvas({
  image,
  masks,
  points,
  boxes,
  mode,
  maskOpacity,
  showMasks,
  hoveredMaskIndex = null,
  onPointAdd,
  onBoxComplete,
  onMaskHover,
}: ImageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);

  // Compute scaled dimensions
  const scaled = image
    ? calculateScaledSize(image.naturalWidth, image.naturalHeight, CANVAS_MAX_WIDTH, CANVAS_MAX_HEIGHT)
    : { scaledWidth: CANVAS_MAX_WIDTH, scaledHeight: CANVAS_MAX_HEIGHT };

  const canvasWidth = scaled.scaledWidth;
  const canvasHeight = scaled.scaledHeight;

  // Scale factors from canvas coords to original image coords
  const scaleX = image ? image.naturalWidth / canvasWidth : 1;
  const scaleY = image ? image.naturalHeight / canvasHeight : 1;

  /** Convert canvas-relative mouse position to image pixel coordinates */
  const canvasToImage = useCallback(
    (cx: number, cy: number) => canvasToImageUtil(cx, cy, scaleX, scaleY),
    [scaleX, scaleY],
  );

  /** Convert image pixel coordinates to canvas coordinates */
  const imageToCanvas = useCallback(
    (ix: number, iy: number) => imageToCanvasUtil(ix, iy, scaleX, scaleY),
    [scaleX, scaleY],
  );

  /** Get mouse position relative to canvas */
  const getCanvasPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    },
    [],
  );

  // Decode mask base64 images and cache them
  useEffect(() => {
    const cache = maskCacheRef.current;
    masks.forEach((m) => {
      if (!cache.has(m.maskBase64)) {
        const img = new Image();
        img.src = `data:image/png;base64,${m.maskBase64}`;
        cache.set(m.maskBase64, img);
      }
    });
  }, [masks]);

  // Main render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // 1. Draw the image
    if (image) {
      ctx.drawImage(image, 0, 0, canvasWidth, canvasHeight);
    }

    // 2. Draw masks
    if (showMasks && masks.length > 0) {
      const cache = maskCacheRef.current;
      masks.forEach((m, i) => {
        const maskImg = cache.get(m.maskBase64);
        if (maskImg && maskImg.complete) {
          const offscreen = document.createElement('canvas');
          offscreen.width = canvasWidth;
          offscreen.height = canvasHeight;
          const offCtx = offscreen.getContext('2d');
          if (offCtx) {
            offCtx.drawImage(maskImg, 0, 0, canvasWidth, canvasHeight);
            offCtx.globalCompositeOperation = 'source-in';
            offCtx.fillStyle = m.color;
            offCtx.fillRect(0, 0, canvasWidth, canvasHeight);
            ctx.save();
            // Highlight hovered mask with higher opacity
            const isHovered = hoveredMaskIndex === i;
            ctx.globalAlpha = isHovered ? Math.min(maskOpacity + 0.3, 1) : maskOpacity;
            ctx.drawImage(offscreen, 0, 0);
            ctx.restore();
          }
        }
      });

      // Draw label for hovered mask
      if (hoveredMaskIndex !== null && masks[hoveredMaskIndex]) {
        const hm = masks[hoveredMaskIndex];
        const [bx1, by1] = hm.bbox;
        const pos = imageToCanvas(bx1, by1);
        const label = hm.label ?? `对象 ${hoveredMaskIndex + 1}`;
        const text = `${label} ${(hm.score * 100).toFixed(1)}%`;

        ctx.save();
        ctx.font = '12px sans-serif';
        const metrics = ctx.measureText(text);
        const px = Math.max(0, pos.x);
        const py = Math.max(16, pos.y - 4);

        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(px, py - 13, metrics.width + 8, 17);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, px + 4, py);
        ctx.restore();
      }
    }

    // 3. Draw point markers
    points.forEach((p) => {
      const pos = imageToCanvas(p.x, p.y);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? FG_COLOR : BG_COLOR;
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // 4. Draw completed boxes
    boxes.forEach((b) => {
      const tl = imageToCanvas(b.x1, b.y1);
      const br = imageToCanvas(b.x2, b.y2);
      ctx.strokeStyle = BOX_STROKE;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.setLineDash([]);
    });

    // 5. Draw in-progress drag box
    if (dragStart && dragCurrent) {
      ctx.strokeStyle = BOX_STROKE;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(
        dragStart.x,
        dragStart.y,
        dragCurrent.x - dragStart.x,
        dragCurrent.y - dragStart.y,
      );
      ctx.setLineDash([]);
    }
  }, [image, masks, points, boxes, showMasks, maskOpacity, canvasWidth, canvasHeight, dragStart, dragCurrent, imageToCanvas, hoveredMaskIndex]);

  // --- Event handlers ---

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image) return;

      if (mode === 'box') {
        e.preventDefault();
        const pos = getCanvasPos(e);
        setDragStart(pos);
        setDragCurrent(pos);
      }
    },
    [image, mode, getCanvasPos],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (mode === 'box' && dragStart) {
        const pos = getCanvasPos(e);
        setDragCurrent(pos);
      }

      // Mask hover detection
      if (masks.length > 0) {
        // Simple bbox-based hover detection
        const pos = getCanvasPos(e);
        const imgPos = canvasToImage(pos.x, pos.y);
        let hoveredIdx: number | null = null;
        for (let i = masks.length - 1; i >= 0; i--) {
          const [bx1, by1, bx2, by2] = masks[i].bbox;
          if (imgPos.x >= bx1 && imgPos.x <= bx2 && imgPos.y >= by1 && imgPos.y <= by2) {
            hoveredIdx = i;
            break;
          }
        }
        onMaskHover(hoveredIdx);
      }
    },
    [mode, dragStart, masks, getCanvasPos, canvasToImage, onMaskHover],
  );

  const handleMouseUp = useCallback(
    (_e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image) return;

      if (mode === 'box' && dragStart && dragCurrent) {
        const startImg = canvasToImage(dragStart.x, dragStart.y);
        const endImg = canvasToImage(dragCurrent.x, dragCurrent.y);

        // Normalize so x1 <= x2, y1 <= y2 and clamp to image bounds
        const box = normalizeBoundingBox(
          startImg.x, startImg.y,
          endImg.x, endImg.y,
          image.naturalWidth, image.naturalHeight,
        );

        // Only register if the box has some area
        if (Math.abs(box.x2 - box.x1) > 2 && Math.abs(box.y2 - box.y1) > 2) {
          onBoxComplete(box);
        }

        setDragStart(null);
        setDragCurrent(null);
      }
    },
    [image, mode, dragStart, dragCurrent, canvasToImage, onBoxComplete],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image || mode !== 'point') return;

      const pos = getCanvasPos(e);
      const imgPos = canvasToImage(pos.x, pos.y);

      // Left click = foreground (label 1)
      onPointAdd({
        x: imgPos.x,
        y: imgPos.y,
        label: 1,
      });
    },
    [image, mode, getCanvasPos, canvasToImage, onPointAdd],
  );

  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image || mode !== 'point') return;
      e.preventDefault();

      const pos = getCanvasPos(e);
      const imgPos = canvasToImage(pos.x, pos.y);

      // Right click = background (label 0)
      onPointAdd({
        x: imgPos.x,
        y: imgPos.y,
        label: 0,
      });
    },
    [image, mode, getCanvasPos, canvasToImage, onPointAdd],
  );

  const handleMouseLeave = useCallback(() => {
    onMaskHover(null);
    if (dragStart) {
      setDragStart(null);
      setDragCurrent(null);
    }
  }, [onMaskHover, dragStart]);

  return (
    <div className="image-canvas-container" style={{ position: 'relative', display: 'inline-block' }}>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        style={{
          border: '1px solid #555',
          cursor: mode === 'point' ? 'crosshair' : mode === 'box' ? 'crosshair' : 'default',
          display: 'block',
        }}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {!image && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: '#888',
            fontSize: '1.1rem',
          }}
        >
          请上传图像
        </div>
      )}
    </div>
  );
}
