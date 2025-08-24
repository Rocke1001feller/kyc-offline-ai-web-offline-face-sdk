import { InferenceSession } from "onnxruntime-web";
import { getModelBytes } from "./model_cache.js";

// These will be replaced by rollup build process
const pkgName = '__PACKAGE_NAME__';
const pkgVersion = '__PACKAGE_VERSION__';

// Register service worker for offline caching (browser only)
if (typeof navigator !== 'undefined' && 'serviceWorker' in navigator && typeof window !== 'undefined') {
  navigator.serviceWorker.register('/sw.js').catch(() => {
    // Silent fail - service worker is optional enhancement
  });
}

async function createSessionWithFallback(modelPath, options = {}) {
  // Input validation
  if (!modelPath || typeof modelPath !== 'string') {
    throw new Error('modelPath must be a non-empty string');
  }

  // Extract filename from path (handle both forward and backward slashes)
  const fileName = modelPath.replace(/\\/g, '/').split('/').pop();
  
  const urls = [
    modelPath,
    `https://unpkg.com/${pkgName}@${pkgVersion}/dist/model/${fileName}`,
    `https://content-deliver.oss-cn-shanghai.aliyuncs.com/${pkgName}/${pkgVersion}/dist/model/${fileName}`,
    `https://cdn-1320106354.cos.ap-shanghai.myqcloud.com/${pkgName}/${pkgVersion}/dist/model/${fileName}`,
    `https://cdn.jsdelivr.net/npm/${pkgName}@${pkgVersion}/dist/model/${fileName}`,
  ];

  try {
    // Try cached approach first (IndexedDB + concurrent download)
    const cacheKey = `${pkgName}:${pkgVersion}:${fileName}`;
    const modelBytes = await getModelBytes(urls, cacheKey);
    return await InferenceSession.create(modelBytes, options);
  } catch (error) {
    // Fallback to original URL-based approach if caching fails
    if (typeof Promise.any === 'function') {
      try {
        return await Promise.any(urls.map(url => InferenceSession.create(url, options)));
      } catch (aggregateError) {
        const firstError = aggregateError.errors?.[0] || aggregateError;
        throw new Error(`Failed to load model: ${firstError.message}`, { cause: firstError });
      }
    }

    // Manual implementation for older browsers
    return new Promise((resolve, reject) => {
      let completedAttempts = 0;
      let resolved = false;
      const errors = [];
      
      urls.forEach((url, index) => {
        InferenceSession.create(url, options)
          .then(session => {
            if (!resolved) {
              resolved = true;
              resolve(session);
            }
          })
          .catch(err => {
            errors[index] = err;
            completedAttempts++;
            
            if (completedAttempts === urls.length && !resolved) {
              const firstError = errors.find(e => e) || new Error('Unknown error');
              reject(new Error(`Failed to load model: ${firstError.message}`, { cause: firstError }));
            }
          });
      });
    });
  }
}

export { createSessionWithFallback };
