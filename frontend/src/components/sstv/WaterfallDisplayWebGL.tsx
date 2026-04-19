"use client";
import React, { useEffect, useRef, useCallback, useMemo, useState } from 'react';

interface WaterfallDisplayProps {
  width: number;
  height: number;
  data: Float32Array;
  minFreq: number;
  maxFreq: number;
  autoScale?: boolean;
  minDb?: number;
  maxDb?: number;
  reverse?: boolean;
  // Zoom/Pan controls
  enableZoom?: boolean;
  enablePan?: boolean;
  onZoomChange?: (zoom: number) => void;
  onPanChange?: (panOffset: number) => void;
}

/**
 * WebGL-accelerated waterfall spectrum display with zoom/pan support.
 * Uses GPU for real-time color mapping and scrolling.
 */
const WaterfallDisplayWebGL: React.FC<WaterfallDisplayProps> = ({
  width,
  height,
  data,
  minFreq,
  maxFreq,
  autoScale = true,
  minDb = -100,
  maxDb = 0,
  reverse = false,
  enableZoom = true,
  enablePan = true,
  onZoomChange,
  onPanChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Zoom/Pan state
  const [zoom, setZoom] = useState<number>(1);
  const [panOffset, setPanOffset] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const lastMouseXRef = useRef<number>(0);

  // Vertex shader source
  const vertexShaderSource = useMemo(() => `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;
    void main() {
      gl_Position = vec4(a_position, 0, 1);
      v_texCoord = a_texCoord;
    }
  `, []);

  // Fragment shader source - color mapping
  const fragmentShaderSource = useMemo(() => `
    precision highp float;
    varying vec2 v_texCoord;
    uniform float u_minDb;
    uniform float u_maxDb;

    // Color palette function
    vec3 dbToColor(float db) {
      // Normalize dB to [0, 1]
      float t = clamp((db - u_minDb) / (u_maxDb - u_minDb), 0.0, 1.0);

      // Heat map color palette
      // Blue -> Cyan -> Green -> Yellow -> Red -> White
      vec3 color;
      if (t < 0.25) {
        // Blue to Cyan
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t / 0.25);
      } else if (t < 0.5) {
        // Cyan to Green
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) / 0.25);
      } else if (t < 0.75) {
        // Green to Yellow
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) / 0.25);
      } else if (t < 0.9) {
        // Yellow to Red
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) / 0.15);
      } else {
        // Red to White
        color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.9) / 0.1);
      }
      return color;
    }

    void main() {
      // v_texCoord.x: frequency (0 = low, 1 = high)
      // v_texCoord.y: time (0 = oldest, 1 = newest)
      float db = v_texCoord.x;
      // Pass dB value through texture
      gl_FragColor = vec4(dbToColor(db), 1.0);
    }
  `, [minDb, maxDb]);

  // Compile shader
  const compileShader = useCallback((gl: WebGLRenderingContext, source: string, type: number) => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) return shader;
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    console.error('Shader compile error:', info);
    return null;
  }, []);

  // Create program
  const createProgram = useCallback((gl: WebGLRenderingContext, vertexSource: string, fragmentSource: string) => {
    const vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);
    if (!vertexShader || !fragmentShader) return null;
    const program = gl.createProgram();
    if (!program) return null;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    const success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success) return program;
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    console.error('Program link error:', info);
    return null;
  }, [compileShader]);

  // Initialize WebGL
  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', {
      antialias: false,
      preserveDrawingBuffer: false,
      alpha: false,
    });

    if (!gl) {
      console.error('WebGL not supported');
      return;
    }

    glRef.current = gl;

    // Create program
    const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    if (!program) return;
    programRef.current = program;

    // Get attribute and uniform locations
    const positionLocation = gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    const minDbLocation = gl.getUniformLocation(program, 'u_minDb');
    const maxDbLocation = gl.getUniformLocation(program, 'u_maxDb');

    // Create buffer
    const buffer = gl.createBuffer();
    bufferRef.current = buffer;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

    // Set up rectangle with texture coordinates
    const vertices = new Float32Array([
      // position     texCoord
      -1, -1,        0, 0,
       1, -1,        1, 0,
      -1,  1,        0, 1,
       1,  1,        1, 1,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // Enable attributes
    const stride = 4 * 4; // 4 floats per vertex (2 position + 2 texCoord)
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, stride, 4 * 2);

    // Create texture
    const texture = gl.createTexture();
    textureRef.current = texture;
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Initialize texture with empty data
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.LUMINANCE,
      data.length,
      1,
      0,
      gl.LUMINANCE,
      gl.FLOAT,
      data
    );

    // Set uniforms
    gl.useProgram(program);
    gl.uniform1f(minDbLocation, minDb);
    gl.uniform1f(maxDbLocation, maxDb);
  }, [vertexShaderSource, fragmentShaderSource, createProgram, data.length, minDb, maxDb]);

  // Update texture with new data
  const updateTexture = useCallback((newData: Float32Array) => {
    const gl = glRef.current;
    const texture = textureRef.current;
    if (!gl || !texture) return;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      newData.length,
      1,
      gl.LUMINANCE,
      gl.FLOAT,
      newData
    );
  }, []);

  // Render frame
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    if (!gl || !program || !texture) return;

    // Set viewport
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use program
    gl.useProgram(program);

    // Update texture
    updateTexture(data);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [width, height, data, updateTexture]);

  // Initialize on mount
  useEffect(() => {
    initWebGL();
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      const gl = glRef.current;
      if (gl) {
        gl.deleteTexture(textureRef.current);
        gl.deleteBuffer(bufferRef.current);
        gl.deleteProgram(programRef.current);
      }
    };
  }, [initWebGL]);

  // Update on data change
  useEffect(() => {
    render();
  }, [render, data]);

  // Update uniforms when minDb/maxDb change
  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    if (!gl || !program) return;

    const minDbLocation = gl.getUniformLocation(program, 'u_minDb');
    const maxDbLocation = gl.getUniformLocation(program, 'u_maxDb');
    gl.useProgram(program);
    gl.uniform1f(minDbLocation, minDb);
    gl.uniform1f(maxDbLocation, maxDb);
  }, [minDb, maxDb]);

  // Zoom/Pan event handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!enableZoom) return;
    e.preventDefault();

    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(1, Math.min(10, zoom * delta));
    setZoom(newZoom);
    onZoomChange?.(newZoom);
  }, [enableZoom, zoom, onZoomChange]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!enablePan) return;
    setIsDragging(true);
    lastMouseXRef.current = e.clientX;
  }, [enablePan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !enablePan) return;

    const deltaX = e.clientX - lastMouseXRef.current;
    lastMouseXRef.current = e.clientX;

    const newPanOffset = panOffset + deltaX * (1000 / zoom);
    const clampedPanOffset = Math.max(-100000, Math.min(100000, newPanOffset));
    setPanOffset(clampedPanOffset);
    onPanChange?.(clampedPanOffset);
  }, [isDragging, enablePan, panOffset, zoom, onPanChange]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Reset zoom/pan
  const handleReset = useCallback(() => {
    setZoom(1);
    setPanOffset(0);
    onZoomChange?.(1);
    onPanChange?.(0);
  }, [onZoomChange, onPanChange]);

  // Adjust frequency range based on pan
  const displayMinFreq = minFreq + panOffset;
  const displayMaxFreq = maxFreq + panOffset;

  return (
    <div
      style={{
        width: `${width}px`,
        height: `${height}px`,
        overflow: 'hidden',
        backgroundColor: '#000',
        border: '1px solid #333',
        borderRadius: '4px',
        position: 'relative',
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: enablePan ? (isDragging ? 'grabbing' : 'grab') : 'default',
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />

      {/* Frequency labels overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: 4,
          left: 4,
          right: 4,
          display: 'flex',
          justifyContent: 'space-between',
          pointerEvents: 'none',
        }}
      >
        <span
          style={{
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: '10px',
            fontFamily: 'monospace',
          }}
        >
          {(displayMinFreq / 1000000).toFixed(3)} MHz
        </span>
        <span
          style={{
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: '10px',
            fontFamily: 'monospace',
          }}
        >
          {(displayMaxFreq / 1000000).toFixed(3)} MHz
        </span>
      </div>

      {/* Zoom/Pan controls */}
      {(enableZoom || enablePan) && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            display: 'flex',
            gap: 4,
            pointerEvents: 'auto',
          }}
        >
          {enableZoom && (
            <>
              <button
                onClick={() => {
                  const newZoom = Math.max(1, zoom * 1.2);
                  setZoom(newZoom);
                  onZoomChange?.(newZoom);
                }}
                style={{
                  width: 24,
                  height: 24,
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: 16,
                  fontWeight: 'bold',
                  color: '#000',
                }}
                title="Zoom in"
              >
                +
              </button>
              <button
                onClick={() => {
                  const newZoom = Math.min(10, zoom / 1.2);
                  setZoom(newZoom);
                  onZoomChange?.(newZoom);
                }}
                style={{
                  width: 24,
                  height: 24,
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: 16,
                  fontWeight: 'bold',
                  color: '#000',
                }}
                title="Zoom out"
              >
                -
              </button>
            </>
          )}
          <button
            onClick={handleReset}
            style={{
              width: 24,
              height: 24,
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: 10,
              fontWeight: 'bold',
              color: '#000',
            }}
            title="Reset zoom/pan"
          >
            R
          </button>
        </div>
      )}

      {/* Zoom level indicator */}
      {zoom > 1 && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            left: 8,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '10px',
            fontFamily: 'monospace',
          }}
        >
          Zoom: {zoom.toFixed(1)}x
        </div>
      )}
    </div>
  );
};

export default WaterfallDisplayWebGL;
