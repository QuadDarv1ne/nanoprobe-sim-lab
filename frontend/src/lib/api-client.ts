import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';

// Export API_BASE for use in other modules (single source of truth)
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Centralized API client with:
 * - Automatic timeout on all requests
 * - Error handling with detailed messages
 * - Request/response interceptors
 * - Retry logic for transient failures
 */

const DEFAULT_TIMEOUT_MS = 10000;
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: DEFAULT_TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
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

  /**
   * Get the underlying axios instance for advanced usage
   */
  getInstance(): AxiosInstance {
    return this.client;
  }

  /**
   * GET request
   */
  async get<T = any>(url: string, config?: any): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  /**
   * POST request
   */
  async post<T = any>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  /**
   * PUT request
   */
  async put<T = any>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  /**
   * DELETE request
   */
  async delete<T = any>(url: string, config?: any): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  /**
   * PATCH request
   */
  async patch<T = any>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
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
