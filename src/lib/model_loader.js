import { InferenceSession } from "onnxruntime-web";

// These will be replaced by rollup build process
const pkgName = '__PACKAGE_NAME__';
const pkgVersion = '__PACKAGE_VERSION__';

async function createSessionWithFallback(modelPath, options = {}) {
  const fileName = modelPath.split('/').pop();
  const fallbacks = [
    modelPath,
    `https://cdn.jsdelivr.net/npm/${pkgName}@${pkgVersion}/dist/model/${fileName}`,
    `https://unpkg.com/${pkgName}@${pkgVersion}/dist/model/${fileName}`
  ];

  for (const url of fallbacks) {
    try {
      return await InferenceSession.create(url, options);
    } catch (e) {
      if (url === fallbacks[fallbacks.length - 1]) throw e;
    }
  }
}

export { createSessionWithFallback };
