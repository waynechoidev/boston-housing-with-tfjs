import { DatasetType, ModelType } from "./constant";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const epochUI = document.getElementById("epoch") as HTMLElement;
const meanLossUI = document.getElementById("mean-loss") as HTMLElement;
const testLossUI = document.getElementById("test-loss") as HTMLElement;
const trainButton = document.getElementById(
  `train-button`
) as HTMLButtonElement;
const modelDropdown = document.getElementById(
  "model-dropdown"
) as HTMLSelectElement;
const graph = document.getElementById("graph") as HTMLElement;

export class Tensors {
  private _numEpochs: number;
  private _batch_size: number;
  private _learningRate: number;

  private _modelType: ModelType;

  private _trainFeatures!: tf.Tensor2D;
  private _trainTarget!: tf.Tensor2D;
  private _testFeatures!: tf.Tensor2D;
  private _testTarget!: tf.Tensor2D;

  constructor({
    numEpochs,
    batchSize,
    learningRate,
  }: {
    numEpochs: number;
    batchSize: number;
    learningRate: number;
  }) {
    this._numEpochs = numEpochs;
    this._batch_size = batchSize;
    this._learningRate = learningRate;
    this._modelType = ModelType.LinearRegressionModel;
  }

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

  get modelLabel() {
    switch (this._modelType) {
      case ModelType.LinearRegressionModel:
        return "Linear Regression Model";
      case ModelType.MultiLayerPerceptronRegressionModel1Hidden:
        return "MLP Regression Model (1 Hidden Layer)";
      case ModelType.MultiLayerPerceptronRegressionModel2Hidden:
        return "MLP Regression Model (2 Hidden Layer)";
      case ModelType.multiLayerPerceptronRegressionModel1HiddenNoSigmoid:
        return "MLP Regression Model (1 Hidden Layer, No Sigmoid)";
    }
  }

  set modelType(type: ModelType) {
    this._modelType = type;
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

  get baseline() {
    const avgPrice = this._trainTarget.mean();

    const diff = this._testTarget.sub(avgPrice);
    const squaredDiff = diff.square();
    const baseline = squaredDiff.mean();

    return baseline.dataSync()[0];
  }

  async trainModel() {
    const model = this.getModel(this._modelType);

    trainButton.disabled = true;
    modelDropdown.disabled = true;

    model.compile({
      optimizer: tf.train.sgd(this._learningRate),
      loss: "meanSquaredError",
    });

    let trainLogs: tf.Logs[] = [];

    await model.fit(this.trainFeatures, this.trainTarget, {
      batchSize: this._batch_size,
      epochs: this._numEpochs,
      // validationSplit: 0.2,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          epochUI!.innerHTML = `Epoch: ${epoch + 1} / ${this._numEpochs}`;
          meanLossUI!.innerHTML = `Mean Loss: ${logs!.loss.toFixed(2)}`;
          trainLogs.push(logs!);
          tfvis.show.history(
            graph,
            trainLogs,
            // ["loss", "val_loss"],
            ["loss"],
            {
              xLabel: "Epoch",
              yLabel: "Mean Loss",
            }
          );
          if (epoch + 1 === this._numEpochs) {
            trainButton.disabled = false;
            modelDropdown.disabled = false;
            meanLossUI!.innerHTML = `Training Set Final Mean Loss: ${logs!.loss.toFixed(
              2
            )}`;
          }
        },
      },
    });

    this.testModel(model);
  }

  testModel(model: tf.Sequential) {
    const predictions = model.predict(this.testFeatures) as tf.Tensor2D;

    const diff = this.testTarget.sub(predictions);
    const squaredDiff = diff.square();
    const meanLoss = squaredDiff.mean();

    testLossUI.innerHTML = `Test Set Mean Loss: ${meanLoss
      .dataSync()[0]
      .toFixed(2)}`;
  }

  cleanResult() {
    epochUI.innerHTML = "";
    meanLossUI.innerHTML = "";
    graph.innerHTML = "";
  }

  // Models

  private get linearRegressionModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [this.numFeatures()], units: 1 }));

    model.summary();
    return model;
  }

  private get multiLayerPerceptronRegressionModel1Hidden() {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [this.numFeatures()],
        units: 50,
        activation: "sigmoid",
        kernelInitializer: "leCunNormal",
      })
    );
    model.add(tf.layers.dense({ units: 1 }));

    model.summary();
    return model;
  }

  private get multiLayerPerceptronRegressionModel2Hidden() {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [this.numFeatures()],
        units: 50,
        activation: "sigmoid",
        kernelInitializer: "leCunNormal",
      })
    );
    model.add(
      tf.layers.dense({
        units: 50,
        activation: "sigmoid",
        kernelInitializer: "leCunNormal",
      })
    );
    model.add(tf.layers.dense({ units: 1 }));

    model.summary();
    return model;
  }

  private get multiLayerPerceptronRegressionModel1HiddenNoSigmoid() {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [this.numFeatures()],
        units: 50,
        // activation: 'sigmoid',
        kernelInitializer: "leCunNormal",
      })
    );
    model.add(tf.layers.dense({ units: 1 }));

    model.summary();
    return model;
  }

  // Private Methods

  private getModel(type: ModelType) {
    switch (type) {
      case ModelType.LinearRegressionModel:
        return this.linearRegressionModel;
      case ModelType.MultiLayerPerceptronRegressionModel1Hidden:
        return this.multiLayerPerceptronRegressionModel1Hidden;
      case ModelType.MultiLayerPerceptronRegressionModel2Hidden:
        return this.multiLayerPerceptronRegressionModel2Hidden;
      case ModelType.multiLayerPerceptronRegressionModel1HiddenNoSigmoid:
        return this.multiLayerPerceptronRegressionModel1HiddenNoSigmoid;
    }
  }

  private numFeatures() {
    return this._trainFeatures.shape[1];
  }

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
