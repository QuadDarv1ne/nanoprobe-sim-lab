import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosRequestConfig } from 'axios';
import { indexedDB, PendingRequest } from './indexeddb';

// Export API_BASE for use in other modules (single source of truth)
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Centralized API client with:
 * - Automatic timeout on all requests
 * - Error handling with detailed messages
 * - Request/response interceptors
 * - Retry logic for transient failures
 * - Offline support with IndexedDB queuing
 */

const DEFAULT_TIMEOUT_MS = 10000;
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;
const OFFLINE_QUEUE_KEY = 'offlineQueue';

class APIClient {
  private client: AxiosInstance;
  private isOnline: boolean = true;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: DEFAULT_TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
    this.setupNetworkListeners();
  }

  private setupInterceptors() {
    // Request interceptor - add auth token if available
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add auth token from localStorage if available
        if (typeof window !== 'undefined') {
          const token = localStorage.getItem('auth_token');
          if (token && config.headers) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        }
        return config;
      },
      (error: AxiosError) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor - handle errors globally
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retryCount?: number };

        // Extract detailed error message
        let errorMessage = 'Unknown error';
        if (error.response) {
          // Server responded with error status
          const status = error.response.status;
          const data = error.response.data as any;
          errorMessage = `HTTP ${status}: ${data?.message || data?.detail || error.response.statusText || 'Server error'}`;
        } else if (error.request) {
          // Request was made but no response received (timeout/network error)
          errorMessage = error.code === 'ECONNABORTED'
            ? `Request timeout after ${DEFAULT_TIMEOUT_MS}ms`
            : 'Network error - check connection';
        } else {
          // Something else happened
          errorMessage = error.message;
        }

        // Log error for debugging (but don't expose sensitive info in production)
        if (process.env.NODE_ENV === 'development') {
          console.error('[API Error]:', {
            message: errorMessage,
            url: originalRequest?.url,
            method: originalRequest?.method,
            status: error.response?.status,
            timestamp: new Date().toISOString()
          });
        }

        // Retry logic for transient errors (5xx, network errors)
        const shouldRetry = (
          error.response?.status &&
          (error.response.status >= 500 || error.response.status === 429)
        ) || !error.response;

        if (shouldRetry && originalRequest && (!originalRequest._retryCount || originalRequest._retryCount < MAX_RETRIES)) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;

          // Wait before retry (exponential backoff)
          const delay = RETRY_DELAY_MS * Math.pow(2, originalRequest._retryCount - 1);
          await new Promise(resolve => setTimeout(resolve, delay));

          return this.client.request(originalRequest);
        }

        // Create enhanced error with context
        const enhancedError = new Error(errorMessage);
        (enhancedError as any).originalError = error;
        (enhancedError as any).status = error.response?.status;
        (enhancedError as any).url = originalRequest?.url;

        return Promise.reject(enhancedError);
      }
    );
  }

  private setupNetworkListeners() {
    if (typeof window !== 'undefined') {
      // Update online status
      const updateOnlineStatus = () => {
        this.isOnline = navigator.onLine;
        if (this.isOnline) {
          // Try to process offline queue when coming online
          this.processOfflineQueue();
        }
      };

      window.addEventListener('online', updateOnlineStatus);
      window.addEventListener('offline', updateOnlineStatus);

      // Set initial status
      this.isOnline = navigator.onLine;
    }
  }

  /**
   * Process the offline queue when connection is restored
   */
  private async processOfflineQueue() {
    if (!this.isOnline) return;

    try {
      const pendingRequests = await indexedDB.getPendingRequests();
      console.log(`[API] Processing ${pendingRequests.length} queued requests`);

      for (const request of pendingRequests) {
        try {
          const response = await this.client.request({
            method: request.method,
            url: request.url,
            data: request.data,
            headers: {
              'Content-Type': 'application/json',
            },
          });

          // Successfully processed, remove from queue
          await indexedDB.deletePendingRequest(request.id);
          console.log(`[API] Successfully processed queued request ${request.id}`);
        } catch (error) {
          console.error(`[API] Failed to process queued request ${request.id}:`, error);
          // Keep in queue for later retry
        }
      }
    } catch (error) {
      console.error('[API] Error processing offline queue:', error);
    }
  }

  /**
   * Queue a request for later processing when offline
   */
  private async queueRequest(request: Omit<PendingRequest, 'id'>): Promise<string> {
    return await indexedDB.addPendingRequest(request);
  }

  /**
   * Get the underlying axios instance for advanced usage
   */
  getInstance(): AxiosInstance {
    return this.client;
  }

  /**
   * GET request with offline support
   */
  async get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
    if (!this.isOnline) {
      // Queue the request for later
      const id = await this.queueRequest({
        method: 'GET',
        url,
        data: undefined,
        timestamp: Date.now(),
        retries: 0,
      });
      throw new Error(`Request queued for offline processing (ID: ${id})`);
    }

    try {
      const response = await this.client.get<T>(url, config);
      return response.data;
    } catch (error) {
      // If network error, queue for later
      if (!navigator.onLine) {
        const id = await this.queueRequest({
          method: 'GET',
          url,
          data: undefined,
          timestamp: Date.now(),
          retries: 0,
        });
        throw new Error(`Request queued for offline processing (ID: ${id})`);
      }
      throw error;
    }
  }

  /**
   * POST request with offline support
   */
  async post<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    if (!this.isOnline) {
      // Queue the request for later
      const id = await this.queueRequest({
        method: 'POST',
        url,
        data,
        timestamp: Date.now(),
        retries: 0,
      });
      throw new Error(`Request queued for offline processing (ID: ${id})`);
    }

    try {
      const response = await this.client.post<T>(url, data, config);
      return response.data;
    } catch (error) {
      // If network error, queue for later
      if (!navigator.onLine) {
        const id = await this.queueRequest({
          method: 'POST',
          url,
          data,
          timestamp: Date.now(),
          retries: 0,
        });
        throw new Error(`Request queued for offline processing (ID: ${id})`);
      }
      throw error;
    }
  }

  /**
   * PUT request with offline support
   */
  async put<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    if (!this.isOnline) {
      // Queue the request for later
      const id = await this.queueRequest({
        method: 'PUT',
        url,
        data,
        timestamp: Date.now(),
        retries: 0,
      });
      throw new Error(`Request queued for offline processing (ID: ${id})`);
    }

    try {
      const response = await this.client.put<T>(url, data, config);
      return response.data;
    } catch (error) {
      // If network error, queue for later
      if (!navigator.onLine) {
        const id = await this.queueRequest({
          method: 'PUT',
          url,
          data,
          timestamp: Date.now(),
          retries: 0,
        });
        throw new Error(`Request queued for offline processing (ID: ${id})`);
      }
      throw error;
    }
  }

  /**
   * DELETE request with offline support
   */
  async delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
    if (!this.isOnline) {
      // Queue the request for later
      const id = await this.queueRequest({
        method: 'DELETE',
        url,
        data: undefined,
        timestamp: Date.now(),
        retries: 0,
      });
      throw new Error(`Request queued for offline processing (ID: ${id})`);
    }

    try {
      const response = await this.client.delete<T>(url, config);
      return response.data;
    } catch (error) {
      // If network error, queue for later
      if (!navigator.onLine) {
        const id = await this.queueRequest({
          method: 'DELETE',
          url,
          data: undefined,
          timestamp: Date.now(),
          retries: 0,
        });
        throw new Error(`Request queued for offline processing (ID: ${id})`);
      }
      throw error;
    }
  }

  /**
   * PATCH request with offline support
   */
  async patch<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    if (!this.isOnline) {
      // Queue the request for later
      const id = await this.queueRequest({
        method: 'PATCH',
        url,
        data,
        timestamp: Date.now(),
        retries: 0,
      });
      throw new Error(`Request queued for offline processing (ID: ${id})`);
    }

    try {
      const response = await this.client.patch<T>(url, data, config);
      return response.data;
    } catch (error) {
      // If network error, queue for later
      if (!navigator.onLine) {
        const id = await this.queueRequest({
          method: 'PATCH',
          url,
          data,
          timestamp: Date.now(),
          retries: 0,
        });
        throw new Error(`Request queued for offline processing (ID: ${id})`);
      }
      throw error;
    }
  }

  /**
   * Update base URL (useful for runtime configuration changes)
   */
  setBaseURL(baseURL: string) {
    this.client.defaults.baseURL = baseURL;
  }
}

// Singleton instance
export const apiClient = new APIClient();

// Export convenience methods
export const { get, post, put, delete: deleteRequest, patch } = apiClient;

export default apiClient;
