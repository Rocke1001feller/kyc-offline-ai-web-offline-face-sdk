export function load_opencv(options?: { basePath?: string }): Promise<void>;
export const cv: any;

// Detection
export function loadDetectionModel(): Promise<any>;
export function loadDetectionModelPath(url: string): Promise<any>;
export function detectFace(session: any, canvasId: string): Promise<any>;
export function detectFaceBase64(session: any, base64Image: string): Promise<any>;

// Liveness
export function loadLivenessModel(): Promise<any>;
export function predictLiveness(session: any, canvasId: string, bbox: any): Promise<any[]>;
export function predictLivenessBase64(session: any, base64Image: string, bbox: any): Promise<any[]>;

// Landmark
export function loadLandmarkModel(): Promise<any>;
export function predictLandmark(session: any, canvasId: string, bbox: any): Promise<number[][]>;
export function predictLandmarkBase64(session: any, base64Image: string, bbox: any): Promise<number[][]>;

// Expression
export function loadExpressionModel(): Promise<any>;
export function predictExpression(session: any, canvasId: string, bbox: any): Promise<any[]>;

// Pose
export function loadPoseModel(): Promise<any>;
export function predictPose(session: any, canvasId: string, bbox: any): Promise<any[]>;
export function softmax(arr: number[]): number[];

// Eye
export function loadEyeModel(): Promise<any>;
export function predictEye(session: any, canvasId: string, landmarks: number[][]): Promise<any[]>;

// Gender
export function loadGenderModel(): Promise<any>;
export function predictGender(session: any, canvasId: string, bbox: any): Promise<any[]>;

// Age
export function loadAgeModel(): Promise<any>;
export function predictAge(session: any, canvasId: string, bbox: any): Promise<any[]>;

// Feature
export function loadFeatureModel(): Promise<any>;
export function extractFeature(session: any, canvasId: string, landmarks: number[][]): Promise<any[]>;
export function extractFeatureBase64(session: any, base64Image: string, landmarks: number[][]): Promise<any[]>;
export function matchFeature(feature1: number[], feature2: number[]): number;
