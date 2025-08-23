var cv = null;

// Internal singleton promise to avoid loading multiple times
let opencvReadyPromise = null;

async function load_opencv(options = {}) {
  const basePath = options.basePath || '..';

  if (cv && typeof cv !== 'undefined') return; // already ready
  if (opencvReadyPromise) return opencvReadyPromise; // loading in progress

  if (!window.WebAssembly) {
    console.log("Your web browser doesn't support WebAssembly.");
    return Promise.reject(new Error('WebAssembly not supported'));
  }

  // If window.cv already exists (preloaded), just resolve
  if (window.cv) {
    cv = window.cv;
    return;
  }

  console.log("loading OpenCv.js");
  opencvReadyPromise = new Promise((resolve, reject) => {
    // Configure Module before loading script
    window.Module = {
      wasmBinaryFile: `${basePath}/js/opencv_js.wasm`,
      preRun: () => {
        // noop
      },
      _main: () => {
        // Called when OpenCV main is ready
        cv = window.cv;
        if (cv) {
          console.log('OpenCV.js is ready.');
          resolve();
        } else {
          reject(new Error('OpenCV loaded but cv is undefined'));
        }
      }
    };

    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.async = true;
    script.src = `${basePath}/js/opencv.js`;
    script.onload = () => {
      console.log('OpenCV.js script loaded.');
      // _main will resolve when cv is ready
    };
    script.onerror = (e) => {
      reject(new Error('Failed to load OpenCV.js script'));
    };
    document.body.appendChild(script);
  });

  return opencvReadyPromise;
}

export {load_opencv, cv}