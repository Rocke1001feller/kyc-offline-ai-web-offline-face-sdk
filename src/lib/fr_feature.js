import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import {cv} from "./load_opencv";
import {createSessionWithFallback} from "./model_loader";

const REFERENCE_FACIAL_POINTS = [
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014],
    [70.72990036, 92.20410156]
]

async function loadFeatureModel() {
  var feature_session = null;
  await createSessionWithFallback("../model/fr_feature.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        feature_session = session
        const input_tensor = new Tensor("float32", new Float32Array(112 * 112 * 3), [1, 3, 112, 112]);
        for (let i = 0; i < 112 * 112 * 3; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = feature_session.run(feeds)
        console.log("initialize the feature session.")
      })
  return feature_session;
}

function convert68pts5pts(landmark) {
    // 输入验证：68个关键点应该有136个数值 (x,y坐标对)
    if (!landmark || landmark.length < 136) {
        throw new Error(`Invalid landmark data: expected at least 136 values, got ${landmark ? landmark.length : 0}`);
    }

    // 68点人脸关键点到5点的标准映射 (基于dlib 68-point model)
    // 左眼中心: 平均左眼内外眼角和上下眼睑的关键点
    var left_eye_x = (landmark[36*2] + landmark[37*2] + landmark[38*2] + landmark[39*2] + 
                     landmark[40*2] + landmark[41*2]) / 6,
        left_eye_y = (landmark[36*2+1] + landmark[37*2+1] + landmark[38*2+1] + landmark[39*2+1] + 
                     landmark[40*2+1] + landmark[41*2+1]) / 6,

        // 右眼中心: 平均右眼内外眼角和上下眼睑的关键点  
        right_eye_x = (landmark[42*2] + landmark[43*2] + landmark[44*2] + landmark[45*2] + 
                      landmark[46*2] + landmark[47*2]) / 6,
        right_eye_y = (landmark[42*2+1] + landmark[43*2+1] + landmark[44*2+1] + landmark[45*2+1] + 
                      landmark[46*2+1] + landmark[47*2+1]) / 6,

        // 鼻尖: 鼻尖点
        nose_x = landmark[30*2], 
        nose_y = landmark[30*2+1],

        // 左嘴角: 左嘴角点
        left_mouth_x = landmark[48*2],
        left_mouth_y = landmark[48*2+1],

        // 右嘴角: 右嘴角点  
        right_mouth_x = landmark[54*2],
        right_mouth_y = landmark[54*2+1];

    const result = [
        [left_eye_x, left_eye_y], 
        [right_eye_x, right_eye_y], 
        [nose_x, nose_y], 
        [left_mouth_x, left_mouth_y],
        [right_mouth_x, right_mouth_y]
    ];

    // 验证转换后的点是否合理
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i][0]) || isNaN(result[i][1])) {
            throw new Error(`Invalid converted point at index ${i}: [${result[i][0]}, ${result[i][1]}]`);
        }
    }

    return result;
}

function getReferenceFacialPoints() {
  let ref5pts = REFERENCE_FACIAL_POINTS;

  return ref5pts;
}

function warpAndCropFace(src,
                       face_pts,
                       ref_pts=null,
                       crop_size=[112, 112]) {
  
  // 参数验证
  if (!face_pts || face_pts.length < 5) {
    throw new Error('Invalid face_pts: expected at least 5 points');
  }
  if (!ref_pts || ref_pts.length < 5) {
    throw new Error('Invalid ref_pts: expected at least 5 points');
  }

  let srcPoints, dstPoints, tfm;
  try {
    // 使用完整的5点进行相似度变换，提高对齐精度
    // 将5个点展开为一维数组 [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
    const srcArray = [];
    const dstArray = [];
    for (let i = 0; i < 5; i++) {
      srcArray.push(face_pts[i][0], face_pts[i][1]);
      dstArray.push(ref_pts[i][0], ref_pts[i][1]);
    }
    
    srcPoints = cv.matFromArray(5, 1, cv.CV_32FC2, srcArray);
    dstPoints = cv.matFromArray(5, 1, cv.CV_32FC2, dstArray);
    
    // 首选：OpenCV 提供的相似度变换接口（有的构建里不存在）
    if (cv.estimateAffinePartial2D && typeof cv.estimateAffinePartial2D === 'function') {
      try {
        tfm = cv.estimateAffinePartial2D(srcPoints, dstPoints);
        if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT) {
          console.log('✅ 使用 OpenCV 原生 estimateAffinePartial2D');
        }
      } catch (e) {
        console.log('⚠️ estimateAffinePartial2D 调用失败，使用手动实现');
        tfm = computeSimilarityTransform(srcArray, dstArray);
      }
    } else {
      // 手动实现 5 点相似度变换（scale+rotation+translation）
      if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT) {
        console.log('ℹ️ OpenCV 不支持 estimateAffinePartial2D，使用手动相似度变换');
      }
      tfm = computeSimilarityTransform(srcArray, dstArray);
    }

    const dsize = new cv.Size(crop_size[0], crop_size[1]);
    const dst = new cv.Mat();
    cv.warpAffine(src, dst, tfm, dsize);

    return dst;
  } finally {
    // 确保内存释放
    [srcPoints, dstPoints, tfm].forEach(mat => {
      if (mat && typeof mat.delete === 'function') {
        mat.delete();
      }
    });
  }
}

function alignFeatureImage(image, landmark) {
  let facePoints = convert68pts5pts(landmark);
  let refPoints = getReferenceFacialPoints();
  let alignImg = warpAndCropFace(image, facePoints, refPoints);
  return alignImg;
}

// 调试和验证函数：可视化人脸对齐质量
function debugAlignment(originalImg, alignedImg, landmarks, canvasId = 'debug-canvas') {
  try {
    // 创建调试canvas
    let debugCanvas = document.getElementById(canvasId);
    if (!debugCanvas) {
      debugCanvas = document.createElement('canvas');
      debugCanvas.id = canvasId;
      debugCanvas.width = 336; // 原图112 + 对齐图112 + 信息区112
      debugCanvas.height = 140; // 增加高度容纳更多信息
      debugCanvas.style.border = '2px solid #ff6b6b';
      debugCanvas.style.position = 'fixed';
      debugCanvas.style.top = '10px';
      debugCanvas.style.right = '10px';
      debugCanvas.style.zIndex = '10000';
      debugCanvas.style.backgroundColor = 'white';
      debugCanvas.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
      debugCanvas.style.borderRadius = '8px';
      debugCanvas.title = '人脸对齐调试：左侧原图+关键点，中间对齐结果+参考点，右侧质量评估';
      document.body.appendChild(debugCanvas);
      
      // 添加关闭按钮
      const closeBtn = document.createElement('button');
      closeBtn.textContent = '×';
      closeBtn.style.position = 'absolute';
      closeBtn.style.top = '5px';
      closeBtn.style.right = '5px';
      closeBtn.style.zIndex = '10001';
      closeBtn.style.background = '#ff6b6b';
      closeBtn.style.color = 'white';
      closeBtn.style.border = 'none';
      closeBtn.style.borderRadius = '50%';
      closeBtn.style.width = '20px';
      closeBtn.style.height = '20px';
      closeBtn.style.cursor = 'pointer';
      closeBtn.onclick = () => {
        debugCanvas.remove();
        closeBtn.remove();
        if (typeof window !== 'undefined') {
          window.DEBUG_FACE_ALIGNMENT = false;
        }
      };
      document.body.appendChild(closeBtn);
    }
    
    const ctx = debugCanvas.getContext('2d');
    ctx.clearRect(0, 0, 336, 140);
    
  // 显示原图 (缩放到112x112) - 使用临时 canvas 避免无效 ID 错误
  const tempOrig = document.createElement('canvas');
  tempOrig.width = originalImg.cols; tempOrig.height = originalImg.rows; tempOrig.style.display='none';
  document.body.appendChild(tempOrig);
  cv.imshow(tempOrig, originalImg);
  ctx.drawImage(tempOrig, 0, 0, 112, 112);
  tempOrig.remove();
    
  // 显示对齐后的图像
  const tempAligned = document.createElement('canvas');
  tempAligned.width = alignedImg.cols; tempAligned.height = alignedImg.rows; tempAligned.style.display='none';
  document.body.appendChild(tempAligned);
  cv.imshow(tempAligned, alignedImg);
  ctx.drawImage(tempAligned, 112, 0, 112, 112);
  tempAligned.remove();
    
    // 在原图上标记5个关键点
    const facePoints = convert68pts5pts(landmarks);
    const refPoints = getReferenceFacialPoints();
    
    // 绘制关键点 - 原图
    ctx.fillStyle = '#ff4757';
    ctx.font = '10px Arial';
    const pointLabels = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角'];
    facePoints.forEach((point, index) => {
      const x = (point[0] / originalImg.cols) * 112;
      const y = (point[1] / originalImg.rows) * 112;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(pointLabels[index], x + 4, y - 4);
    });
    
    // 绘制参考点 - 对齐图
    ctx.fillStyle = '#3742fa';
    refPoints.forEach((point, index) => {
      const x = 112 + point[0];
      const y = point[1];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(pointLabels[index], x + 4, y - 4);
    });
    
    // 计算对齐质量指标
    const quality = calculateAlignmentQuality(facePoints, refPoints);
    
    // 绘制信息面板
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(224, 0, 112, 112);
    ctx.strokeStyle = '#dee2e6';
    ctx.strokeRect(224, 0, 112, 112);
    
    // 显示质量指标
    ctx.fillStyle = '#212529';
    ctx.font = '11px monospace';
    ctx.fillText('对齐质量评估', 228, 15);
    ctx.font = '9px monospace';
    
    ctx.fillStyle = quality.eyeDistance > 0.9 ? '#28a745' : '#dc3545';
    ctx.fillText(`眼距比: ${quality.eyeDistance.toFixed(3)}`, 228, 30);
    
    ctx.fillStyle = quality.faceSymmetry > 0.9 ? '#28a745' : '#dc3545';  
    ctx.fillText(`对称性: ${quality.faceSymmetry.toFixed(3)}`, 228, 45);
    
    ctx.fillStyle = quality.poseStability > 0.8 ? '#28a745' : '#dc3545';
    ctx.fillText(`姿态: ${quality.poseStability.toFixed(3)}`, 228, 60);
    
    ctx.fillStyle = quality.overall >= 0.8 ? '#28a745' : quality.overall >= 0.6 ? '#ffc107' : '#dc3545';
    ctx.fillText(`总分: ${quality.overall.toFixed(3)}`, 228, 75);
    
    // 显示建议
    ctx.fillStyle = '#6c757d';
    ctx.font = '8px Arial';
    if (quality.overall < 0.6) {
      ctx.fillText('建议调整关键点', 228, 90);
      ctx.fillText('检测算法', 228, 100);
    } else if (quality.overall < 0.8) {
      ctx.fillText('对齐质量一般', 228, 90);
    } else {
      ctx.fillText('对齐质量良好', 228, 90);
    }
    
    // 添加标签
    ctx.fillStyle = '#343a40';
    ctx.font = 'bold 10px Arial';
    ctx.fillText('原图+关键点', 5, 125);
    ctx.fillText('对齐+参考点', 117, 125);
    ctx.fillText('质量分析', 229, 125);
    
  } catch (error) {
    console.warn('Debug alignment visualization failed:', error);
  }
}

// 计算对齐质量指标
function calculateAlignmentQuality(facePoints, refPoints) {
  try {
    // 1. 眼距比例检查
    const faceEyeDist = Math.sqrt(
      Math.pow(facePoints[1][0] - facePoints[0][0], 2) + 
      Math.pow(facePoints[1][1] - facePoints[0][1], 2)
    );
    const refEyeDist = Math.sqrt(
      Math.pow(refPoints[1][0] - refPoints[0][0], 2) + 
      Math.pow(refPoints[1][1] - refPoints[0][1], 2)
    );
    const eyeDistanceRatio = Math.min(faceEyeDist, refEyeDist) / Math.max(faceEyeDist, refEyeDist);
    
    // 2. 面部对称性检查 (左右眼到鼻尖距离)
    const leftEyeToNose = Math.sqrt(
      Math.pow(facePoints[0][0] - facePoints[2][0], 2) + 
      Math.pow(facePoints[0][1] - facePoints[2][1], 2)
    );
    const rightEyeToNose = Math.sqrt(
      Math.pow(facePoints[1][0] - facePoints[2][0], 2) + 
      Math.pow(facePoints[1][1] - facePoints[2][1], 2)
    );
    const symmetry = Math.min(leftEyeToNose, rightEyeToNose) / Math.max(leftEyeToNose, rightEyeToNose);
    
    // 3. 姿态稳定性 (眼睛水平程度)
    const eyeAngle = Math.abs(Math.atan2(
      facePoints[1][1] - facePoints[0][1], 
      facePoints[1][0] - facePoints[0][0]
    ));
    const poseStability = Math.max(0, 1 - eyeAngle / (Math.PI / 6)); // 30度内为满分
    
    // 综合评分
    const overall = (eyeDistanceRatio * 0.4 + symmetry * 0.3 + poseStability * 0.3);
    
    return {
      eyeDistance: eyeDistanceRatio,
      faceSymmetry: symmetry,
      poseStability: poseStability,
      overall: overall
    };
  } catch (error) {
    return {
      eyeDistance: 0,
      faceSymmetry: 0, 
      poseStability: 0,
      overall: 0
    };
  }
}

// 手动 5 点相似度变换（基于最小二乘法）：输入展开数组 [x1,y1,...,x5,y5]
function computeSimilarityTransform(srcArray, dstArray) {
  try {
    const n = 5;
    const srcPts = []; const dstPts = [];
    for (let i=0;i<n;i++) {
      srcPts.push([srcArray[2*i], srcArray[2*i+1]]);
      dstPts.push([dstArray[2*i], dstArray[2*i+1]]);
    }
    
    // 使用最小二乘法求解相似度变换参数
    // 相似度变换矩阵: [a, -b, tx; b, a, ty] 
    // 即 x' = a*x - b*y + tx, y' = b*x + a*y + ty
    
    let sum_x = 0, sum_y = 0, sum_u = 0, sum_v = 0;
    let sum_xx = 0, sum_yy = 0, sum_xy = 0;
    let sum_ux = 0, sum_uy = 0, sum_vx = 0, sum_vy = 0;
    
    for (let i = 0; i < n; i++) {
      const x = srcPts[i][0];
      const y = srcPts[i][1];
      const u = dstPts[i][0];
      const v = dstPts[i][1];
      
      sum_x += x; sum_y += y; sum_u += u; sum_v += v;
      sum_xx += x*x; sum_yy += y*y; sum_xy += x*y;
      sum_ux += u*x; sum_uy += u*y; sum_vx += v*x; sum_vy += v*y;
    }
    
    // 构建并求解线性方程组
    const SX = sum_xx + sum_yy;
    const SY = sum_x*sum_x + sum_y*sum_y;
    const A1 = sum_ux + sum_vy;
    const A2 = sum_vx - sum_uy;
    const B1 = sum_u;
    const B2 = sum_v;
    
    const denom = SX*n - SY;
    if (Math.abs(denom) < 1e-10) {
      // 退化情况：使用仿射变换
      console.warn('退化到前3点仿射变换');
      const srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, srcArray.slice(0, 6));
      const dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, dstArray.slice(0, 6));
      const tfm = cv.getAffineTransform(srcTri, dstTri);
      srcTri.delete();
      dstTri.delete();
      return tfm;
    }
    
    // 求解相似度变换参数
    const a = (A1*n - B1*sum_x - B2*sum_y) / denom;
    const b = (A2*n - B2*sum_x + B1*sum_y) / denom;
    const tx = (B1 - a*sum_x + b*sum_y) / n;
    const ty = (B2 - b*sum_x - a*sum_y) / n;
    
    // 验证参数合理性
    const scale = Math.sqrt(a*a + b*b);
    if (scale < 0.1 || scale > 10) {
      console.warn('相似度变换scale异常:', scale, '，回退到仿射变换');
      const srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, srcArray.slice(0, 6));
      const dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, dstArray.slice(0, 6));
      const tfm = cv.getAffineTransform(srcTri, dstTri);
      srcTri.delete();
      dstTri.delete();
      return tfm;
    }
    
    // 构建仿射变换矩阵 [2x3]
    const transform = cv.matFromArray(2, 3, cv.CV_32FC1, [a, -b, tx, b, a, ty]);
    
    // 调试输出
    if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT) {
      const scale = Math.sqrt(a*a + b*b);
      const angle = Math.atan2(b, a) * 180 / Math.PI;
      console.log('🔧 相似度变换参数:', {
        缩放: scale.toFixed(3),
        旋转角度: angle.toFixed(1) + '°',
        平移: `(${tx.toFixed(1)}, ${ty.toFixed(1)})`
      });
    }
    
    return transform;
    
  } catch (e) {
    console.warn('相似度变换计算失败:', e, '使用单位变换');
    return cv.matFromArray(2, 3, cv.CV_32FC1, [1,0,0,0,1,0]);
  }
}

function preprocessFeature(image) {
  var rows = image.rows,
      cols = image.cols;

  // 参数验证
  if (rows !== 112 || cols !== 112) {
    console.warn(`Expected 112x112 image, got ${rows}x${cols}. This may affect model performance.`);
  }

  var img_data = ndarray(new Float32Array(rows * cols * 3), [rows, cols, 3]);

  // 优化：减少函数调用，提高性能
  for (var y = 0; y < rows; y++) {
    for (var x = 0; x < cols; x++) {
      let pixel = image.ucharPtr(y, x);
      
      // OpenCV.js使用BGR格式，需要转换为RGB
      // 模型训练时通常使用RGB格式和[-1,1]归一化范围
      var pixel_value_b = (pixel[0] - 127.5) / 127.5; // B -> R (in RGB)
      var pixel_value_g = (pixel[1] - 127.5) / 127.5; // G -> G  
      var pixel_value_r = (pixel[2] - 127.5) / 127.5; // R -> B (in RGB)

      // 按RGB顺序存储 (注意BGR到RGB的转换)
      img_data.set(y, x, 0, pixel_value_r); // R channel
      img_data.set(y, x, 1, pixel_value_g); // G channel  
      img_data.set(y, x, 2, pixel_value_b); // B channel
    }
  }

  // 转换为NCHW格式 [1, 3, 112, 112]
  var preprocessed = ndarray(new Float32Array(3 * rows * cols), [1, 3, rows, cols]);

  ops.assign(preprocessed.pick(0, 0, null, null), img_data.pick(null, null, 0)); // R
  ops.assign(preprocessed.pick(0, 1, null, null), img_data.pick(null, null, 1)); // G
  ops.assign(preprocessed.pick(0, 2, null, null), img_data.pick(null, null, 2)); // B

  return preprocessed;
}

async function extractFeatureImage(session, img, landmarks) {
  const result = [];
  const input_tensor = new Tensor("float32", new Float32Array(112 * 112 * 3), [1, 3, 112, 112]);
  
  try {
    for (let i = 0; i < landmarks.length; i++) {
      let face_img = null;
      try {
        face_img = alignFeatureImage(img, landmarks[i]);
        
        // 可选：启用调试可视化
        if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT) {
          debugAlignment(img, face_img, landmarks[i]);
          
          // 增强的控制台日志
          const facePoints = convert68pts5pts(landmarks[i]);
          const quality = calculateAlignmentQuality(facePoints, getReferenceFacialPoints());
          console.log(`🎯 人脸 #${i+1} 对齐质量:`, {
            总分: quality.overall.toFixed(3),
            眼距比: quality.eyeDistance.toFixed(3), 
            对称性: quality.faceSymmetry.toFixed(3),
            姿态: quality.poseStability.toFixed(3),
            评级: quality.overall >= 0.8 ? '优秀✨' : quality.overall >= 0.6 ? '良好👍' : '需改进⚠️'
          });
        }
        
        const input_image = preprocessFeature(face_img);
        
        // 复用tensor，提高性能
        input_tensor.data.set(input_image.data);
        const feeds = {"input": input_tensor};

        const output_map = await session.run(feeds);
        // 兼容不同模型输出Key：优先使用 'output'，否则取首个key
        let tensorData = null;
        if (output_map) {
          if (output_map.output && output_map.output.data) {
            tensorData = output_map.output.data;
          } else {
            const keys = Object.keys(output_map);
            if (keys.length > 0 && output_map[keys[0]] && output_map[keys[0]].data) {
              tensorData = output_map[keys[0]].data;
            }
          }
        }
        if (!tensorData) {
            console.warn('未找到特征向量输出，输出map keys=', output_map ? Object.keys(output_map) : 'null');
        } else {
            // 复制为独立的 Float32Array，避免后续复用覆盖
            const featureVector = new Float32Array(tensorData.length);
            featureVector.set(tensorData);
            result.push(featureVector);
        }
        
      } catch (error) {
        console.error(`❌ 处理第${i}个人脸时出错:`, error);
        // 继续处理其他人脸
      } finally {
        // 确保释放face_img内存
        if (face_img && typeof face_img.delete === 'function') {
          face_img.delete();
        }
      }
    }
    
    // 批处理完成日志
    if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT && result.length > 0) {
      console.log(`✅ 特征提取完成: ${result.length}个人脸，特征维度: ${result[0].length}`);
    }
    
  } finally {
    // 注意：img由调用方负责释放，这里不删除
    // img.delete(); // 移除这行，避免重复释放
  }
  
  return result;
}

async function extractFeature(session, canvas_id, landmarks) {
  let img = null;
  try {
    img = cv.imread(canvas_id);
    const result = await extractFeatureImage(session, img, landmarks);
    return result;
  } finally {
    if (img && typeof img.delete === 'function') {
      img.delete();
    }
  }
}

async function extractFeatureBase64(session, base64Image, landmarks) {
  let image = new Image();
  let img = null;
  
  try {
    image.src = base64Image;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
    });

    img = cv.imread(image);
    const result = await extractFeatureImage(session, img, landmarks);
    return result;
  } finally {
    if (img && typeof img.delete === 'function') {
      img.delete();
    }
  }
}

function matchFeature(feature1, feature2) {
  if (!feature1 || !feature2 || feature1.length !== feature2.length) {
    console.warn('特征向量格式不匹配:', {
      feature1Length: feature1?.length || 0,
      feature2Length: feature2?.length || 0
    });
    return 0;
  }

  const vectorSize = feature1.length;
  
  // 创建副本，避免修改原数组
  const f1 = new Float32Array(feature1);
  const f2 = new Float32Array(feature2);

  // 计算均值并中心化
  let f1Sum = 0, f2Sum = 0;
  for (let i = 0; i < vectorSize; i++) {
    const meanVal = (f1[i] + f2[i]) / 2;
    f1[i] -= meanVal;
    f2[i] -= meanVal;
    
    f1Sum += f1[i] * f1[i];
    f2Sum += f2[i] * f2[i];
  }

  // 检查数值稳定性
  if (f1Sum < 1e-12 || f2Sum < 1e-12) {
    console.warn('特征向量方差过小，可能为零向量');
    return 0;
  }

  // 归一化并计算余弦相似度
  const norm1 = Math.sqrt(f1Sum);
  const norm2 = Math.sqrt(f2Sum);
  
  let dotProduct = 0;
  for (let i = 0; i < vectorSize; i++) {
    f1[i] = f1[i] / norm1;
    f2[i] = f2[i] / norm2;
    dotProduct += f1[i] * f2[i];
  }

  // 限制在[-1,1]范围内，避免数值误差
  return Math.max(-1, Math.min(1, dotProduct));
}



// 公开版本：人脸对齐质量验证函数（供外部调用）
function validateAlignmentQuality(originalImg, alignedImg, landmarks, canvasId = 'quality-canvas') {
  try {
    // 创建调试canvas
    let debugCanvas = document.getElementById(canvasId);
    if (!debugCanvas) {
      debugCanvas = document.createElement('canvas');
      debugCanvas.id = canvasId;
      debugCanvas.width = 336; // 原图112 + 对齐图112 + 信息区112
      debugCanvas.height = 140; // 增加高度容纳更多信息
      debugCanvas.style.border = '2px solid #ff6b6b';
      debugCanvas.style.position = 'fixed';
      debugCanvas.style.top = '10px';
      debugCanvas.style.right = '10px';
      debugCanvas.style.zIndex = '10000';
      debugCanvas.style.backgroundColor = 'white';
      debugCanvas.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
      debugCanvas.style.borderRadius = '8px';
      debugCanvas.title = '人脸对齐调试：左侧原图+关键点，中间对齐结果+参考点，右侧质量评估';
      document.body.appendChild(debugCanvas);
      
      // 添加关闭按钮
      const closeBtn = document.createElement('button');
      closeBtn.textContent = '×';
      closeBtn.style.position = 'absolute';
      closeBtn.style.top = '5px';
      closeBtn.style.right = '5px';
      closeBtn.style.zIndex = '10001';
      closeBtn.style.background = '#ff6b6b';
      closeBtn.style.color = 'white';
      closeBtn.style.border = 'none';
      closeBtn.style.borderRadius = '50%';
      closeBtn.style.width = '20px';
      closeBtn.style.height = '20px';
      closeBtn.style.cursor = 'pointer';
      closeBtn.onclick = () => {
        debugCanvas.remove();
        closeBtn.remove();
        if (typeof window !== 'undefined') {
          window.DEBUG_FACE_ALIGNMENT = false;
        }
      };
      document.body.appendChild(closeBtn);
    }
    
    const ctx = debugCanvas.getContext('2d');
    ctx.clearRect(0, 0, 336, 140);
    
  // 显示原图 (缩放到112x112) - 使用临时 canvas 避免无效 ID 错误
  const tempOrig = document.createElement('canvas');
  tempOrig.width = originalImg.cols; tempOrig.height = originalImg.rows; tempOrig.style.display='none';
  document.body.appendChild(tempOrig);
  cv.imshow(tempOrig, originalImg);
  ctx.drawImage(tempOrig, 0, 0, 112, 112);
  tempOrig.remove();
    
  // 显示对齐后的图像
  const tempAligned = document.createElement('canvas');
  tempAligned.width = alignedImg.cols; tempAligned.height = alignedImg.rows; tempAligned.style.display='none';
  document.body.appendChild(tempAligned);
  cv.imshow(tempAligned, alignedImg);
  ctx.drawImage(tempAligned, 112, 0, 112, 112);
  tempAligned.remove();
    
    // 在原图上标记5个关键点
    const facePoints = convert68pts5pts(landmarks);
    const refPoints = getReferenceFacialPoints();
    
    // 绘制关键点 - 原图
    ctx.fillStyle = '#ff4757';
    ctx.font = '10px Arial';
    const pointLabels = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角'];
    facePoints.forEach((point, index) => {
      const x = (point[0] / originalImg.cols) * 112;
      const y = (point[1] / originalImg.rows) * 112;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(pointLabels[index], x + 4, y - 4);
    });
    
    // 绘制参考点 - 对齐图
    ctx.fillStyle = '#3742fa';
    refPoints.forEach((point, index) => {
      const x = 112 + point[0];
      const y = point[1];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(pointLabels[index], x + 4, y - 4);
    });
    
    // 计算对齐质量指标
    const quality = calculateAlignmentQuality(facePoints, refPoints);
    
    // 绘制信息面板
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(224, 0, 112, 112);
    ctx.strokeStyle = '#dee2e6';
    ctx.strokeRect(224, 0, 112, 112);
    
    // 显示质量指标
    ctx.fillStyle = '#212529';
    ctx.font = '11px monospace';
    ctx.fillText('对齐质量评估', 228, 15);
    ctx.font = '9px monospace';
    
    ctx.fillStyle = quality.eyeDistance > 0.9 ? '#28a745' : '#dc3545';
    ctx.fillText(`眼距比: ${quality.eyeDistance.toFixed(3)}`, 228, 30);
    
    ctx.fillStyle = quality.faceSymmetry > 0.9 ? '#28a745' : '#dc3545';  
    ctx.fillText(`对称性: ${quality.faceSymmetry.toFixed(3)}`, 228, 45);
    
    ctx.fillStyle = quality.poseStability > 0.8 ? '#28a745' : '#dc3545';
    ctx.fillText(`姿态: ${quality.poseStability.toFixed(3)}`, 228, 60);
    
    ctx.fillStyle = quality.overall >= 0.8 ? '#28a745' : quality.overall >= 0.6 ? '#ffc107' : '#dc3545';
    ctx.fillText(`总分: ${quality.overall.toFixed(3)}`, 228, 75);
    
    // 显示建议
    ctx.fillStyle = '#6c757d';
    ctx.font = '8px Arial';
    if (quality.overall < 0.6) {
      ctx.fillText('建议调整关键点', 228, 90);
      ctx.fillText('检测算法', 228, 100);
    } else if (quality.overall < 0.8) {
      ctx.fillText('对齐质量一般', 228, 90);
    } else {
      ctx.fillText('对齐质量良好', 228, 90);
    }
    
    // 添加标签
    ctx.fillStyle = '#343a40';
    ctx.font = 'bold 10px Arial';
    ctx.fillText('原图+关键点', 5, 125);
    ctx.fillText('对齐+参考点', 117, 125);
    ctx.fillText('质量分析', 229, 125);
    
  } catch (error) {
    console.warn('Debug alignment visualization failed:', error);
  }
}

// 公开版本：特征一致性验证函数（供外部调用）
async function validateFeatureConsistency(session, canvasId, landmarks, iterations = 10) {
  console.log(`🧪 开始特征一致性测试 (${iterations}次迭代)...`);
  const features = [];
  
  try {
    for (let i = 0; i < iterations; i++) {
      const result = await extractFeature(session, canvasId, landmarks);
      if (result && result.length > 0) {
        const vec = result[0];
        if (vec && vec.length > 0) {
          // 确保是独立的数组副本
          features.push(Array.from(vec));
        }
      }
      
      // 小延迟避免过快调用
      if (i < iterations - 1) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
    
    if (features.length < 2) {
      console.warn('❌ 特征提取失败，无法进行一致性测试');
      return null;
    }
    
    // 计算特征向量间的相似度统计
    const similarities = [];
    const firstFeature = features[0];
    
    console.log(`📊 计算 ${features.length} 个特征向量间的相似度...`);
    for (let i = 1; i < features.length; i++) {
      const similarity = matchFeature(firstFeature, features[i]);
      similarities.push(similarity);
      
      if (typeof window !== 'undefined' && window.DEBUG_FACE_ALIGNMENT) {
        console.log(`   特征1 vs 特征${i+1}: ${similarity.toFixed(4)}`);
      }
    }
    
    // 计算统计指标
    const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    const minSimilarity = Math.min(...similarities);
    const maxSimilarity = Math.max(...similarities);
    const variance = similarities.reduce((acc, val) => acc + Math.pow(val - avgSimilarity, 2), 0) / similarities.length;
    const stdDev = Math.sqrt(variance);
    
    // 评估标准：高质量系统的特征应该高度一致
    const isHighlyConsistent = stdDev < 0.005 && avgSimilarity > 0.99;
    const isConsistent = stdDev < 0.02 && avgSimilarity > 0.95;
    
    const report = {
      iterations,
      featuresExtracted: features.length,
      featureDimension: firstFeature.length,
      avgSimilarity: parseFloat(avgSimilarity.toFixed(6)),
      minSimilarity: parseFloat(minSimilarity.toFixed(6)),
      maxSimilarity: parseFloat(maxSimilarity.toFixed(6)),
      standardDeviation: parseFloat(stdDev.toFixed(6)),
      isHighlyConsistent,
      isConsistent,
      qualityLevel: isHighlyConsistent ? '优秀' : isConsistent ? '良好' : '需改进'
    };
    
    console.log('📋 特征一致性测试报告:', {
      '质量等级': report.qualityLevel + (isHighlyConsistent ? ' 🌟' : isConsistent ? ' ✅' : ' ⚠️'),
      '平均相似度': report.avgSimilarity,
      '标准差': report.standardDeviation,
      '相似度范围': `[${report.minSimilarity}, ${report.maxSimilarity}]`,
      '特征维度': report.featureDimension
    });
    
    if (!report.isConsistent) {
      console.warn('⚠️ 特征提取一致性较差，可能原因:');
      console.warn('   1. 人脸对齐算法不稳定 (检查对齐质量评分)');
      console.warn('   2. 关键点检测存在抖动 (观察关键点是否稳定)');
      console.warn('   3. 图像预处理数值精度问题');
      console.warn('   4. ONNX模型推理存在随机性');
    } else if (report.isHighlyConsistent) {
      console.log('✨ 特征提取质量优秀，系统稳定性很高！');
    } else {
      console.log('👍 特征提取一致性良好');
    }
    
    return report;
    
  } catch (error) {
    console.error('❌ 特征一致性测试失败:', error);
    return null;
  }
}

export {loadFeatureModel, extractFeature, extractFeatureBase64, matchFeature, debugAlignment, validateAlignmentQuality, validateFeatureConsistency}