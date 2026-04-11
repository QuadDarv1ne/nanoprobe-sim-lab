/**
 * NASA API React Hooks
 *
 * Хуки для работы с NASA API:
 * - useAPOD - Astronomy Picture of the Day
 * - useMarsPhotos - Photos from Mars rovers
 * - useAsteroids - Near Earth Objects
 * - useEarthImagery - Earth images from EPIC
 * - useNASAImageLibrary - NASA image library search
 * - useNaturalEvents - EONET natural events
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api/v1/nasa';

// ==========================================
// Types
// ==========================================

interface APOD {
  date: string;
  explanation: string;
  hdurl?: string;
  media_type: 'image' | 'video';
  service_version: string;
  title: string;
  url: string;
  copyright?: string;
  cached?: boolean;
}

interface MarsPhoto {
  id: number;
  img_src: string;
  earth_date: string;
  rover: {
    name: string;
    landing_date: string;
    launch_date: string;
    status: 'active' | 'completed';
  };
  camera: {
    full_name: string;
    name: string;
  };
}

interface Asteroid {
  id: number;
  name: string;
  diameter: {
    estimated_diameter_min: number;
    estimated_diameter_max: number;
  };
  close_approach_data: Array<{
    close_approach_date: string;
    miss_distance: {
      kilometers: string;
    };
  }>;
  is_potentially_hazardous_asteroid: boolean;
}

interface NaturalEvent {
  id: string;
  title: string;
  description: string;
  category: string;
  geometry: Array<{
    date: string;
    coordinates: [number, number];
  }>;
}

// ==========================================
// Generic fetch hook
// ==========================================

type NASAQueryParams = Record<string, string | number | boolean | undefined>;

function useNASAData<T>(endpoint: string, params?: NASAQueryParams) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cached, setCached] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${API_BASE}${endpoint}`, { params });
      setData(response.data);
      setCached(response.data.cached || false);
    } catch (err: unknown) {
      const errorMessage =
        axios.isAxiosError(err)
          ? (err.response?.data?.detail || err.message || 'NASA API error')
          : (err instanceof Error ? err.message : 'Unknown error');
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [endpoint, JSON.stringify(params || {})]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, cached, refetch: fetchData };
}

// ==========================================
// APOD Hook
// ==========================================

export function useAPOD(date?: string) {
  const { data, loading, error, cached, refetch } = useNASAData<APOD>('/apod', { date });

  return {
    apod: data,
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Mars Photos Hook
// ==========================================

export function useMarsPhotos(params?: {
  sol?: number;
  earth_date?: string;
  camera?: string;
  rover?: string;
  page?: number;
  per_page?: number;
}) {
  const { data, loading, error, cached, refetch } = useNASAData<{
    photos: MarsPhoto[];
    total_count: number;
  }>('/mars/photos', params);

  return {
    photos: data?.photos || [],
    totalCount: data?.total_count || 0,
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Asteroids Hook
// ==========================================

export function useAsteroids(params?: {
  start_date?: string;
  end_date?: string;
  page?: number;
  per_page?: number;
}) {
  const { data, loading, error, cached, refetch } = useNASAData<{
    near_earth_objects: Record<string, Asteroid[]>;
    element_count: number;
  }>('/asteroids/feed', params);

  return {
    asteroids: data?.near_earth_objects || {},
    elementCount: data?.element_count || 0,
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Natural Events Hook
// ==========================================

export function useNaturalEvents(params?: {
  status?: 'open' | 'closed';
  days?: number;
  limit?: number;
}) {
  const { data, loading, error, cached, refetch } = useNASAData<{
    events: NaturalEvent[];
  }>('/events/natural', params);

  return {
    events: data?.events || [],
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Earth Imagery Hook
// ==========================================

interface EarthImageryData {
  date?: string;
  identifier?: string;
  caption?: string;
  image?: string;
  thumbnail?: string;
  lat?: number;
  lon?: number;
  terrain?: string;
  time?: string;
  acquisition_terms?: string;
}

export function useEarthImagery(params?: {
  date?: string;
  lat?: number;
  lon?: number;
}) {
  const { data, loading, error, cached, refetch } = useNASAData<EarthImageryData | EarthImageryData[]>('/earth/imagery', params);

  return {
    images: data ? (Array.isArray(data) ? data : [data]) : [],
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Image Library Search Hook
// ==========================================

export function useNASAImageLibrary(params: {
  query: string;
  media_type?: 'image' | 'video' | 'audio';
  page?: number;
  page_size?: number;
}) {
  const { data, loading, error, cached, refetch } = useNASAData<{
    collection: {
      items: Array<{
        data: {
          nasa_id: string;
          title: string;
          description: string;
          center: string;
          date_created: string;
        };
        links: { href: string }[];
      }>;
    };
  }>('/image-library/search', params);

  return {
    results: data?.collection?.items || [],
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// Mars Rover Manifest Hook
// ==========================================

export function useMarsRovers() {
  const { data, loading, error, cached, refetch } = useNASAData<{
    rovers: Array<{
      id: number;
      name: string;
      landing_date: string;
      launch_date: string;
      status: string;
      max_sol: number;
      total_photos: number;
    }>;
  }>('/mars/rovers');

  return {
    rovers: data?.rovers || [],
    loading,
    error,
    cached,
    refetch,
  };
}

// ==========================================
// NASA API Health Check
// ==========================================

export function useNASAHealth() {
  const [status, setStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>('unknown');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE}/health`);
        setStatus(response.data.status as 'healthy' | 'unhealthy');
      } catch {
        setStatus('unhealthy');
      } finally {
        setLoading(false);
      }
    };

    checkHealth();

    // Check every 5 minutes
    const interval = setInterval(checkHealth, 300000);
    return () => clearInterval(interval);
  }, []);

  return { status, loading };
}
