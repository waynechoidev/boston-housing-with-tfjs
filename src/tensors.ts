import { DatasetType } from "./constant";
import * as tf from "@tensorflow/tfjs";

export class Tensors {
  private _trainFeatures!: tf.Tensor2D;
  private _trainTarget!: tf.Tensor2D;
  private _testFeatures!: tf.Tensor2D;
  private _testTarget!: tf.Tensor2D;

  constructor() {}

  init(dataset: DatasetType) {
    const rawTrainFeatures = tf.tensor2d(dataset.trainFeatures);
    const rawTestFeatures = tf.tensor2d(dataset.testFeatures);

    this._trainTarget = tf.tensor2d(dataset.trainTarget);
    this._testTarget = tf.tensor2d(dataset.testTarget);

    let { dataMean, dataStd } = this.determineMeanAndStddev(rawTrainFeatures);

    this._trainFeatures = this.normalizeTensor(
      rawTrainFeatures,
      dataMean,
      dataStd
    );
    this._testFeatures = this.normalizeTensor(
      rawTestFeatures,
      dataMean,
      dataStd
    );
  }

  get trainFeatures() {
    return this._trainFeatures;
  }
  get trainTarget() {
    return this._trainTarget;
  }
  get testFeatures() {
    return this._testFeatures;
  }
  get testTarget() {
    return this._testTarget;
  }

  //   get baseline() {
  //     const avgPrice = this.mean(this._trainTarget);

  //     const diff = this.sub(this._testTarget, avgPrice);
  //     const squaredDiff = this.square(diff);
  //     const baseline = this.mean(squaredDiff);

  //     return baseline;
  //   }

  private determineMeanAndStddev(data: tf.Tensor2D) {
    const dataMean = data.mean(0);
    const diffFromMean = data.sub(dataMean);
    const squaredDiffFromMean = diffFromMean.square();
    const variance = squaredDiffFromMean.mean(0);
    const dataStd = variance.sqrt();
    return { dataMean, dataStd };
  }

  private normalizeTensor(
    data: tf.Tensor2D,
    dataMean: tf.Tensor<tf.Rank>,
    dataStd: tf.Tensor<tf.Rank>
  ): tf.Tensor2D {
    return data.sub(dataMean).div(dataStd);
  }
}
