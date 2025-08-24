/**
 * Simple model cache using IndexedDB with concurrent-safe downloading
 * Falls back gracefully when IndexedDB is unavailable
 */

const memoryCache = new Map();
const downloadPromises = new Map();
const DB_NAME = 'onnx-models';
const STORE_NAME = 'models';

let dbPromise = null;

function openDB() {
  if (!('indexedDB' in window)) return Promise.resolve(null);
  
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE_NAME);
      };
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null); // Graceful fallback
    });
  }
  
  return dbPromise;
}

async function getFromIndexedDB(key) {
  try {
    const db = await openDB();
    if (!db) return null;
    
    return new Promise((resolve) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const request = transaction.objectStore(STORE_NAME).get(key);
      
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function putToIndexedDB(key, value) {
  try {
    const db = await openDB();
    if (!db) return;
    
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const request = transaction.objectStore(STORE_NAME).put(value, key);
    
    // Ignore storage quota errors - graceful degradation
    request.onerror = () => {};
  } catch {
    // Silent fail - cache is optional
  }
}

async function downloadModel(urls) {
  // Use Promise.any for concurrent downloads with fallback for older browsers
  const downloadAttempts = urls.map(url => 
    fetch(url).then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
      return response.arrayBuffer();
    })
  );

  if (typeof Promise.any === 'function') {
    return await Promise.any(downloadAttempts);
  }

  // Manual Promise.any implementation
  return new Promise((resolve, reject) => {
    let remainingAttempts = downloadAttempts.length;
    let lastError;

    downloadAttempts.forEach(promise => {
      promise
        .then(resolve)
        .catch(error => {
          lastError = error;
          remainingAttempts--;
          if (remainingAttempts === 0) reject(lastError);
        });
    });
  });
}

/**
 * Get model bytes with caching - returns Uint8Array for onnxruntime-web
 * @param {string[]} urls - Fallback URLs to try
 * @param {string} cacheKey - Unique cache key (should include version)
 * @returns {Promise<Uint8Array>} Model bytes ready for InferenceSession.create
 */
export async function getModelBytes(urls, cacheKey) {
  // 1. Check memory cache first (fastest)
  if (memoryCache.has(cacheKey)) {
    return memoryCache.get(cacheKey);
  }

  // 2. Ensure single download per model (concurrent safety)
  if (downloadPromises.has(cacheKey)) {
    return downloadPromises.get(cacheKey);
  }

  const downloadPromise = (async () => {
    // 3. Try IndexedDB cache
    const cached = await getFromIndexedDB(cacheKey);
    if (cached instanceof ArrayBuffer) {
      const bytes = new Uint8Array(cached);
      memoryCache.set(cacheKey, bytes);
      return bytes;
    }

    // 4. Download with concurrent fallbacks
    const arrayBuffer = await downloadModel(urls);
    const bytes = new Uint8Array(arrayBuffer);

    // 5. Cache for next time
    putToIndexedDB(cacheKey, arrayBuffer); // Fire and forget
    memoryCache.set(cacheKey, bytes);

    return bytes;
  })();

  downloadPromises.set(cacheKey, downloadPromise);

  try {
    return await downloadPromise;
  } finally {
    downloadPromises.delete(cacheKey);
  }
}
