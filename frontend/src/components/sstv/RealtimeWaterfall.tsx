"use client";

import React, { useEffect, useRef, useCallback, useState } from 'react';
import WaterfallDisplayWebGL from '@/components/sstv/WaterfallDisplayWebGL';
import { API_BASE } from '@/lib/config';

interface WaterfallDisplayProps {
  width?: number;
  height?: number;
  frequency?: number;
  sampleRate?: number;
}

/**
 * Real-time spectrum waterfall display using WebSocket.
 * Fetches spectrum data from API and displays using WebGL.
 */
const RealtimeWaterfall: React.FC<WaterfallDisplayProps> = ({
  width = 800,
  height = 400,
  frequency = 145.800,
  sampleRate = 2400000,
}) => {
  const [spectrumData, setSpectrumData] = useState<Float32Array | null>(null);
  const [minFreq, setMinFreq] = useState<number>(frequency * 1e6 - 50000);
  const [maxFreq, setMaxFreq] = useState<number>(frequency * 1e6 + 50000);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const animationRef = useRef<number | null>(null);
  const dataArrayRef = useRef<Float32Array>(new Float32Array(1024));

  // Connect to WebSocket for real-time spectrum data
  const connectWebSocket = useCallback(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.host.replace(':3000', ':8000');
    const wsUrl = `${wsProtocol}://${host}/api/v1/sstv/ws/stream`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Waterfall WebSocket connected');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'spectrum' && msg.power_db) {
            // Update frequency range if provided
            if (msg.min_freq !== undefined) {
              setMinFreq(msg.min_freq);
            }
            if (msg.max_freq !== undefined) {
              setMaxFreq(msg.max_freq);
            }

            // Convert to Float32Array if needed
            const powerData = new Float32Array(msg.power_db);
            setSpectrumData(powerData);

            // Update data array for continuous display
            if (powerData.length !== dataArrayRef.current.length) {
              dataArrayRef.current = new Float32Array(powerData.length);
            }
            dataArrayRef.current.set(powerData);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('Waterfall WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;
        // Attempt reconnection after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws.close();
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      setTimeout(connectWebSocket, 5000);
    }
  }, []);

  // Fallback: fetch spectrum data via HTTP if WebSocket unavailable
  const fetchSpectrumFallback = useCallback(async () => {
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/sstv/spectrum?frequency=${frequency}&duration=0.1`
      );
      if (res.ok) {
        const data = await res.json();
        if (data.power_db) {
          const powerData = new Float32Array(data.power_db);
          setSpectrumData(powerData);
        }
      }
    } catch (e) {
      // Silently fail, WebSocket will retry
    }
  }, [frequency]);

  // Initialize connection
  useEffect(() => {
    connectWebSocket();

    // Fallback polling if WebSocket fails
    const fallbackInterval = setInterval(fetchSpectrumFallback, 100);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      clearInterval(fallbackInterval);
    };
  }, [connectWebSocket, fetchSpectrumFallback]);

  // Update frequency range when frequency prop changes
  useEffect(() => {
    const freqHz = frequency * 1e6;
    setMinFreq(freqHz - 50000);
    setMaxFreq(freqHz + 50000);
  }, [frequency]);

  return (
    <div className="space-y-4">
      {/* Connection status */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'
          }`}
        />
        <span className="text-sm text-muted-foreground">
          {isConnected ? 'Подключено' : 'Подключение...'}
        </span>
      </div>

      {/* Waterfall display */}
      <div className="relative">
        <WaterfallDisplayWebGL
          width={width}
          height={height}
          data={spectrumData || new Float32Array(1024).fill(-100)}
          minFreq={minFreq}
          maxFreq={maxFreq}
          autoScale={true}
          minDb={-100}
          maxDb={0}
        />

        {/* Loading indicator */}
        {!spectrumData && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2" />
              <p className="text-white text-sm">Загрузка спектра...</p>
            </div>
          </div>
        )}
      </div>

      {/* Frequency info */}
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{(minFreq / 1e6).toFixed(3)} MHz</span>
        <span>{frequency.toFixed(3)} MHz</span>
        <span>{(maxFreq / 1e6).toFixed(3)} MHz</span>
      </div>
    </div>
  );
};

export default RealtimeWaterfall;
