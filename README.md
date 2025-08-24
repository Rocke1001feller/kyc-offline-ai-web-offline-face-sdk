# KYC Offline AI --- Web Offline Face SDK

🎯 **极致简洁的离线人脸识别 SDK**，基于 ONNX Runtime 和 OpenCV.js，支持浏览器端实时推理。

[![npm version](https://img.shields.io/npm/v/kyc-offline-ai-web-offline-face-sdk.svg)](https://www.npmjs.com/package/kyc-offline-ai-web-offline-face-sdk)
[![ES Module](https://img.shields.io/badge/ESM-ready-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔗 相关链接

- 📦 [NPM Package](https://www.npmjs.com/package/kyc-offline-ai-web-offline-face-sdk)
- 🚀 [在线演示 Demo](https://kyc-offline-ai-web-offline.pages.dev)
- 📱 [Android SDK 演示](https://kyc-offline-ai.pages.dev/)
- 💻 [GitHub 仓库](https://github.com/Rocke1001feller/kyc-offline-ai-web-offline-face-sdk.git)

## ✨ 特性

- 🚀 **纯浏览器运行** - 无需服务器，完全离线
- ⚡ **智能缓存系统** - IndexedDB + Service Worker 双重缓存，首次加载后离线可用
- 🎯 **功能全面** - 人脸检测、活体检测、关键点、表情、年龄、性别识别
- 🌐 **多源并发加载** - 多CDN同时竞速，最快响应优先
- ⚡ **极致性能** - ONNX Runtime + WebAssembly 优化
- 🔧 **现代模块输出** - ESM + CJS + UMD 三格式支持
- 📦 **零依赖** - 仅依赖必要的运行时库
- 🎨 **API 极简** - 函数式设计，简洁优雅

## 🚀 快速开始

### 安装

```bash
npm i kyc-offline-ai-web-offline-face-sdk
```

### 基础使用

```javascript
import {
  load_opencv,
  loadDetectionModel, detectFace,
  loadLivenessModel, predictLiveness,
} from 'kyc-offline-ai-web-offline-face-sdk';

// 1. 初始化 OpenCV
await load_opencv();

// 2. 加载模型
const detectionSession = await loadDetectionModel();
const livenessSession = await loadLivenessModel();

// 3. 人脸检测
const faceResult = await detectFace(detectionSession, 'canvas-id');
if (faceResult.size > 0) {
  // 获取第一张人脸的边界框
  const bbox = [
    faceResult.bbox.get(0, 0), // x1
    faceResult.bbox.get(0, 1), // y1
    faceResult.bbox.get(0, 2), // x2
    faceResult.bbox.get(0, 3)  // y2
  ];
  
  // 4. 活体检测
  const livenessResult = await predictLiveness(livenessSession, 'canvas-id', faceResult.bbox);
  console.log('活体分数:', livenessResult[0][4]);
}
```

### 静态资源部署

SDK 需要以下静态资源，请确保在运行时可访问：

```bash
# 复制到 public 目录
cp -r node_modules/kyc-offline-ai-web-offline-face-sdk/dist/model public/
cp -r node_modules/kyc-offline-ai-web-offline-face-sdk/dist/js public/
cp node_modules/kyc-offline-ai-web-offline-face-sdk/dist/sw.js public/
```

**智能缓存系统**（v1.0.6+）：
- **首次加载**：多CDN并发竞速，自动选择最快源
- **后续加载**：IndexedDB + Service Worker 双重缓存，秒级加载
- **离线模式**：Service Worker 拦截请求，完全离线可用
- **自动失效**：版本升级时缓存自动清理更新
## API 文档

**1. 人脸检测**
```javascript
const session = await loadDetectionModel();
const faceResult = await detectFace(session, 'canvas-id');
// 支持Base64图像
const faceResult = await detectFaceBase64(session, base64Image);
```

**2. 活体检测**
```javascript
const session = await loadLivenessModel();
const livenessResult = await predictLiveness(session, 'canvas-id', bbox);
// 支持Base64图像
const livenessResult = await predictLivenessBase64(session, base64Image, bbox);
```

**3. 特征提取**
```javascript
const session = await loadFeatureModel();
const features = await extractFeature(session, 'canvas-id', landmarks);
// 支持Base64图像
const features = await extractFeatureBase64(session, base64Image, landmarks);
```

其他API：`loadDetectionModelPath`、`loadLandmarkModel`、`predictLandmark`、`predictLandmarkBase64`、`loadAgeModel`、`predictAge`、`loadGenderModel`、`predictGender`、`loadExpressionModel`、`predictExpression`、`loadEyeModel`、`predictEye`、`loadPoseModel`、`predictPose`、`matchFeature`

**4. 其他实用函数**
```javascript
// 从自定义路径加载检测模型
const session = await loadDetectionModelPath('path/to/model.onnx');

// 人脸特征匹配
const similarity = matchFeature(feature1, feature2);
console.log('相似度:', similarity);
```

## 🎨 完整示例

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/kyc-offline-ai-web-offline-face-sdk@1.0.6/dist/index.umd.js"></script>
</head>
<body>
    <video id="video" autoplay></video>
    <canvas id="canvas" style="display:none;"></canvas>
    
    <script>
        const { detectFace, loadDetectionModel, loadLivenessModel, predictLiveness } = window.KycOfflineAiWebOfflineFaceSdk;
        
        (async function() {
            // 获取摄像头
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            // 预加载模型（首次会从多CDN并发下载并缓存）
            const detectionSession = await loadDetectionModel();
            const livenessSession = await loadLivenessModel();
            console.log('模型加载完成，后续使用将从本地缓存瞬时加载');
            
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                setInterval(async () => {
                    ctx.drawImage(video, 0, 0);
                    
                    // 检测人脸（后续调用将使用缓存，毫秒级响应）
                    const faceResult = await detectFace(detectionSession, canvas);
                    if (faceResult.size > 0) {
                        const livenessResult = await predictLiveness(livenessSession, canvas, faceResult.bbox);
                        console.log('活体检测结果:', livenessResult[0][4] > 0.5 ? '真人' : '假脸');
                    }
                }, 1000);
            };
        })();
    </script>
</body>
</html>
```

## 📋 支持的功能

| 功能 | 函数 | 说明 |
|------|------|------|
| 人脸检测 | `detectFace` | 检测图像中的人脸位置 |
| 活体检测 | `predictLiveness` | 判断是否为真实人脸 |
| 关键点检测 | `predictLandmark` | 68个面部关键点 |
| 表情识别 | `predictExpression` | 7种基本表情 |
| 年龄识别 | `predictAge` | 估算年龄 |
| 性别识别 | `predictGender` | 男性/女性分类 |
| 眼部检测 | `predictEye` | 睁眼/闭眼状态 |
| 姿态检测 | `predictPose` | 头部姿态角度 |
| 特征提取 | `extractFeature` | 人脸特征向量 |
| 特征匹配 | `matchFeature` | 计算特征相似度 |

## 🚀 版本更新

### v1.0.6 (2025-08-24)

**🆕 重大功能**
- **智能缓存系统**: IndexedDB + Service Worker 双重缓存机制
- **多源并发加载**: 多CDN同时竞速下载，最快响应优先
- **离线优先体验**: 首次加载后完全离线可用，零网络延迟
- **并发安全机制**: 同一模型多次同时请求时共享下载任务

**⚡ 性能提升** 
- **首次加载**: 多源并发，比单一CDN平均快30-50%
- **后续加载**: 本地缓存，加载时间从秒级降至毫秒级
- **离线模式**: Service Worker拦截，完全无网络依赖
- **内存优化**: 智能缓存清理，避免内存泄漏

**🔧 技术实现**
- **多层缓存**: Memory → IndexedDB → Service Worker → 网络请求
- **优雅降级**: 缓存系统失效时自动回退到原URL加载方式
- **版本管理**: 缓存key包含版本号，升级时自动失效重新下载
- **环境适配**: 自动检测浏览器支持度，Node.js环境下跳过浏览器特性

### v1.0.5 (2025-08-23)

**🆕 新增功能**
- **智能CDN降级**: 实现 `../model/` → `cdn.jsdelivr.net` → `unpkg.com` 三级降级机制，提升模型加载可靠性
- **Base64活体检测增强**: `predictLivenessBase64(session, base64Image, bbox)` 新增 bbox 参数支持

**🔧 优化改进** 
- **OpenCV加载重构**: `load_opencv()` 采用 Promise + 单例模式，防止重复加载和竞态条件
- **错误处理完善**: 模型加载失败时自动尝试CDN源，提升网络环境适应性

## 🛠️ 技术栈

- **推理引擎**: ONNX Runtime Web
- **图像处理**: OpenCV.js
- **数值计算**: ndarray + ndarray-ops
- **构建工具**: Rollup
- **包管理**: pnpm workspaces

## 📦 文件结构

```
kyc-offline-ai-web-offline-face-sdk/
├── dist/
│   ├── index.esm.js       # ES Module 版本
│   ├── index.cjs          # CommonJS 版本
│   ├── index.umd.js       # UMD 版本（浏览器直接引用）
│   ├── sw.js              # Service Worker（离线缓存）
│   ├── model/             # ONNX 模型文件（构建复制）
│   └── js/                # OpenCV.js 资源（构建复制）
├── src/
│   ├── lib/               # 功能模块（JS）
│   ├── model/             # ONNX 模型文件（源）
│   └── index.js           # 入口文件（ API 导出）
└── package.json
```

## 🔧 开发指南

### 本地开发

```bash
# 安装依赖
pnpm install

# 构建
pnpm build
```

### 模型部署

SDK 使用相对路径加载资源：
- OpenCV.js：`../js/opencv.js` 和 `../js/opencv_js.wasm`
- ONNX 模型：`../model/*.onnx`
- Service Worker：`/sw.js`（用于离线缓存）

**智能加载机制** (v1.0.6+)：
1. **多源并发**: 同时从多个CDN源下载，最快响应优先
2. **本地缓存**: IndexedDB 持久化存储，版本化管理
3. **Service Worker**: HTTP级别拦截，完全离线支持
4. **优雅降级**: 缓存失败时自动回退到网络加载

## ⚠️ 注意事项

- **首次加载**: 模型文件约45MB，首次会并发下载并缓存到本地
- **后续使用**: 从IndexedDB和Service Worker缓存加载，毫秒级响应
- **浏览器兼容**: 需要支持WASM、IndexedDB和Service Worker，推荐Chrome 90+
- **离线使用**: Service Worker需要通过HTTP/HTTPS协议访问才能正常工作
- **存储配额**: IndexedDB缓存会占用浏览器存储空间，通常无需担心

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 🙏 致谢

- [ONNX Runtime](https://onnxruntime.ai/) - 高性能推理引擎
- [OpenCV.js](https://opencv.org/opencv.js/) - 浏览器端计算机视觉库

## 📄 开源协议

MIT License - 可自由使用于商业和开源项目

**- 📱 [Android SDK 商业化项目请查看](https://kyc-offline-ai.pages.dev/)**

---

**让人脸识别在浏览器中飞起来！** 🚀✨
