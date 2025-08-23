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
- 🎯 **功能全面** - 人脸检测、活体检测、关键点、表情、年龄、性别识别
- ⚡ **极致性能** - ONNX Runtime + WebAssembly 优化
- 🌐 **智能降级** - 本地 → CDN 自动降级，确保高可用性
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
```
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
    <script src="https://unpkg.com/kyc-offline-ai-web-offline-face-sdk@1.0.5/dist/index.umd.js"></script>
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
            
            // 预加载模型
            const detectionSession = await loadDetectionModel();
            const livenessSession = await loadLivenessModel();
            console.log('模型加载完成');
            
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                setInterval(async () => {
                    ctx.drawImage(video, 0, 0);
                    
                    // 检测人脸
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

**CDN 降级机制** (v1.0.5+)：
1. 优先从本地 `../model/` 加载
2. 失败时尝试 `cdn.jsdelivr.net` CDN
3. 再失败时尝试 `unpkg.com` CDN

## ⚠️ 注意事项

- **模型文件**: 总大小约45MB，首次加载需要时间
- **浏览器兼容**: 需要支持WASM和Canvas API，推荐Chrome 90+
- **CORS配置**: CDN模型文件需要正确配置跨域头

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
