/**
 * Service Worker for offline model caching
 * Intercepts .onnx requests and serves from cache when available
 */

const CACHE_NAME = 'onnx-models-v1';

// Install event - prepare cache
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME));
  self.skipWaiting(); // Activate immediately
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(cacheNames => 
      Promise.all(
        cacheNames
          .filter(name => name.startsWith('onnx-models-') && name !== CACHE_NAME)
          .map(name => caches.delete(name))
      )
    )
  );
  self.clients.claim(); // Take control immediately
});

// Fetch event - intercept model requests
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle .onnx model files
  if (!url.pathname.endsWith('.onnx')) {
    return; // Let browser handle normally
  }

  event.respondWith(
    (async () => {
      try {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(request);

        if (cached) {
          return cached; // Serve from cache
        }

        // Not in cache - fetch and cache
        const response = await fetch(request);
        
        // Only cache successful responses
        if (response.ok && response.status === 200) {
          // Clone before caching (response body can only be read once)
          cache.put(request, response.clone());
        }

        return response;
      } catch (error) {
        // Network error - try cache as last resort
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(request);
        
        if (cached) {
          return cached;
        }
        
        // Re-throw if no cache available
        throw error;
      }
    })()
  );
});
