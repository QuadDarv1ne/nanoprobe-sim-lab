import React, { useEffect, useRef, useCallback } from 'react';

interface WaterfallDisplayProps {
  width: number;
  height: number;
  data: Float32Array;
  minFreq: number;
  maxFreq: number;
}

/**
 * Maps a dBFS value to an RGB color string.
 * Color scale: blue=-100, green=-60, yellow=-30, red=-10, white=0
 */
function dbfsToColor(value: number): string {
  const clamped = Math.max(-100, Math.min(0, value));

  if (clamped <= -100) return 'rgb(0, 0, 255)';
  if (clamped <= -60) {
    const t = (clamped + 100) / 40;
    return `rgb(0, ${Math.round(255 * t)}, ${Math.round(255 * (1 - t))})`;
  }
  if (clamped <= -30) {
    const t = (clamped + 60) / 30;
    return `rgb(${Math.round(255 * t)}, 255, 0)`;
  }
  if (clamped <= -10) {
    const t = (clamped + 30) / 20;
    return `rgb(255, ${Math.round(255 * (1 - t))}, 0)`;
  }
  if (clamped <= 0) {
    const t = (clamped + 10) / 10;
    return `rgb(255, ${Math.round(255 * t)}, ${Math.round(255 * t)})`;
  }
  return 'rgb(255, 255, 255)';
}

/**
 * Parse CSS color string to [r, g, b] tuple.
 * Handles rgb(r,g,b) format produced by dbfsToColor.
 */
function parseColor(rgb: string): [number, number, number] {
  const match = rgb.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!match) return [0, 0, 0];
  return [parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10)];
}

const WATERFALL_LINE_HEIGHT = 2;

const WaterfallDisplay: React.FC<WaterfallDisplayProps> = ({
  width,
  height,
  data,
  minFreq,
  maxFreq,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const drawWaterfall = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const numBins = data.length;
    const canvasWidth = canvas.width;
    const lineHeight = WATERFALL_LINE_HEIGHT;

    // Shift existing image down to make room for new line
    const existingData = ctx.getImageData(0, 0, canvasWidth, canvas.height);
    ctx.putImageData(existingData, 0, lineHeight);

    // Draw new spectrum line at the top
    const imageData = ctx.createImageData(canvasWidth, lineHeight);
    const pixels = imageData.data;

    for (let x = 0; x < canvasWidth; x++) {
      const dataIndex = Math.floor((x / canvasWidth) * numBins);
      const dbValue = data[Math.min(dataIndex, numBins - 1)];
      const [r, g, b] = parseColor(dbfsToColor(dbValue));

      for (let y = 0; y < lineHeight; y++) {
        const index = (y * canvasWidth + x) * 4;
        pixels[index] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
        pixels[index + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw frequency axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '10px monospace';
    const numLabels = 5;
    const freqRange = maxFreq - minFreq;

    for (let i = 0; i <= numLabels; i++) {
      const freq = minFreq + (freqRange * i) / numLabels;
      const xPos = (i / numLabels) * canvasWidth;
      const label = freq >= 1000
        ? `${(freq / 1000).toFixed(2)} GHz`
        : `${freq.toFixed(2)} MHz`;
      ctx.fillText(label, xPos, canvas.height - 4);
    }
  }, [data, minFreq, maxFreq]);

  useEffect(() => {
    drawWaterfall();
  }, [drawWaterfall]);

  return (
    <div
      ref={containerRef}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        overflowY: 'auto',
        overflowX: 'hidden',
        backgroundColor: '#000',
        border: '1px solid #333',
        borderRadius: '4px',
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ display: 'block' }}
        /* WebGL fallback note: Canvas 2D is primary; ctx.getContext('webgl')
           could be used for GPU-accelerated color mapping on large datasets */
      />
    </div>
  );
};

export default WaterfallDisplay;
