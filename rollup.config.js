import { readFileSync } from 'fs';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import json from '@rollup/plugin-json';
import replace from '@rollup/plugin-replace';
import copy from 'rollup-plugin-copy';
import terser from '@rollup/plugin-terser';

const production = !process.env.ROLLUP_WATCH;

// Read package.json for version info
const packageJson = JSON.parse(readFileSync('./package.json', 'utf8'));
const version = packageJson.version;

// Base plugins used across builds
const basePlugins = [
  resolve({ browser: true, preferBuiltins: false }),
  commonjs(),
  json(),
  replace({
    '__PACKAGE_NAME__': packageJson.name,
    '__PACKAGE_VERSION__': version,
    preventAssignment: true
  }),
  copy({
    targets: [
      { src: 'src/model/*', dest: 'dist/model' },
      { src: 'src/js/opencv*.js', dest: 'dist/js' },
      { src: 'src/js/opencv_js.wasm', dest: 'dist/js' },
      // ship types to dist
      { src: 'src/types/index.d.ts', dest: 'dist', rename: () => 'index.d.ts' },
      // copy service worker for caching (only to root)
      { src: 'src/js/sw.js', dest: 'dist' },
    ],
    copyOnce: true,
  }),
  production && terser(),
];

// ESM & CJS build (keep externals)
const esmCjsConfig = {
  input: 'src/index.js',
  output: [
    { file: 'dist/index.esm.js', format: 'esm', sourcemap: true, inlineDynamicImports: true },
    { file: 'dist/index.cjs', format: 'cjs', sourcemap: true, inlineDynamicImports: true },
  ],
  plugins: basePlugins,
  external: ['onnxruntime-web', 'ndarray', 'ndarray-ops'],
};

// UMD build (bundle deps for script-tag usage)
const umdConfig = {
  input: 'src/index.js',
  output: {
    file: 'dist/index.umd.js',
    format: 'umd',
    name: 'KycOfflineAiWebOfflineFaceSdk',
    sourcemap: true,
    inlineDynamicImports: true,
  },
  plugins: basePlugins,
  external: [],
};

export default [esmCjsConfig, umdConfig];
