/**
 * IndexedDB utility for Nanoprobe Sim Lab
 * Handles offline storage and synchronization
 */

export interface PendingRequest {
  id: string;
  method: 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'GET';
  url: string;
  data?: unknown;
  timestamp: number;
  retries: number;
}

class IndexedDB {
  private dbName = 'nanoprobeLabDB';
  private version = 1;
  private db: IDBDatabase | null = null;
  private pendingStoreName = 'pendingRequests';

  async init(): Promise<void> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const request = window.indexedDB.open(this.dbName, this.version);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(this.pendingStoreName)) {
          db.createObjectStore(this.pendingStoreName, { keyPath: 'id' });
        }
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onerror = () => {
        reject(request.error);
      };
    });
  }

  private getStore(mode: IDBTransactionMode = 'readonly') {
    if (!this.db) {
      throw new Error('IndexedDB not initialized');
    }
    return this.db.transaction(this.pendingStoreName, mode).objectStore(this.pendingStoreName);
  }

  async addPendingRequest(request: Omit<PendingRequest, 'id'>): Promise<string> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
      // Return a dummy ID for SSR
      return 'dummy-id';
    }
    const id = Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
    const pendingRequest: PendingRequest = {
      id,
      ...request,
      timestamp: Date.now(),
      retries: 0,
    };

    return new Promise((resolve, reject) => {
      try {
        const store = this.getStore('readwrite');
        const requestStore = store.add(pendingRequest);
        requestStore.onsuccess = () => resolve(id);
        requestStore.onerror = () => reject(requestStore.error);
      } catch (error) {
        reject(error);
      }
    });
  }

  async getPendingRequests(): Promise<PendingRequest[]> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
      return Promise.resolve([]);
    }
    return new Promise((resolve, reject) => {
      try {
        const store = this.getStore();
        const requestStore = store.getAll();
        requestStore.onsuccess = () => resolve(requestStore.result);
        requestStore.onerror = () => reject(requestStore.error);
      } catch (error) {
        reject(error);
      }
    });
  }

  async deletePendingRequest(id: string): Promise<void> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      try {
        const store = this.getStore('readwrite');
        const requestStore = store.delete(id);
        requestStore.onsuccess = () => resolve();
        requestStore.onerror = () => reject(requestStore.error);
      } catch (error) {
        reject(error);
      }
    });
  }

  async clearPendingRequests(): Promise<void> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      try {
        const store = this.getStore('readwrite');
        const requestStore = store.clear();
        requestStore.onsuccess = () => resolve();
        requestStore.onerror = () => reject(requestStore.error);
      } catch (error) {
        reject(error);
      }
    });
  }
}

export const indexedDB = new IndexedDB();

// Initialize on module load (only in browser)
if (typeof window !== 'undefined') {
  indexedDB.init().catch(console.error);
}
