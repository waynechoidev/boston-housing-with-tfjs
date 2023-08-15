export const BASE_URL =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/";
export const TRAIN_FEATURES_FN = "train-data.csv";
export const TRAIN_TARGET_FN = "train-target.csv";
export const TEST_FEATURES_FN = "test-data.csv";
export const TEST_TARGET_FN = "test-target.csv";

export type DatasetType = {
  trainFeatures: number[][];
  trainTarget: number[][];
  testFeatures: number[][];
  testTarget: number[][];
};

export type ChartDataType = { epoch: number; meanLoss: number };
