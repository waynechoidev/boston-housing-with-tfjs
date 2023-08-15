import * as Papa from "papaparse";
import {
  BASE_URL,
  DatasetType,
  TEST_FEATURES_FN,
  TEST_TARGET_FN,
  TRAIN_FEATURES_FN,
  TRAIN_TARGET_FN,
} from "./constant";

export class BostonHousingDataset {
  private _trainFeatures?: number[][];
  private _trainTarget?: number[][];
  private _testFeatures?: number[][];
  private _testTarget?: number[][];

  constructor() {}

  async loadData() {
    const loadProcess = document.getElementById("load-process");
    const loadButton = document.getElementById(
      "load-button"
    ) as HTMLButtonElement;
    if (!loadProcess) console.error("There is no element to show process");

    loadProcess!.innerHTML = "It is loading now.";
    loadButton!.disabled = true;

    [
      this._trainFeatures,
      this._trainTarget,
      this._testFeatures,
      this._testTarget,
    ] = await Promise.all([
      this.loadCsv(TRAIN_FEATURES_FN),
      this.loadCsv(TRAIN_TARGET_FN),
      this.loadCsv(TEST_FEATURES_FN),
      this.loadCsv(TEST_TARGET_FN),
    ]);

    this.shuffle(this._trainFeatures, this._trainTarget);
    this.shuffle(this._testFeatures, this._testTarget);
    if (
      !!this._testFeatures ||
      !!this._trainTarget ||
      !!this._testFeatures ||
      !!this._testTarget
    ) {
      loadProcess!.innerHTML = "It is done!";
    } else {
      loadButton!.disabled = false;
    }
  }

  get dataset(): DatasetType {
    return {
      trainFeatures: this._trainFeatures ?? [],
      trainTarget: this._trainTarget ?? [],
      testFeatures: this._testFeatures ?? [],
      testTarget: this._testTarget ?? [],
    };
  }

  // Private methods
  private async loadCsv(filename: string) {
    return new Promise<number[][]>((resolve) => {
      const url = `${BASE_URL}${filename}`;
      Papa.parse<object>(url, {
        download: true,
        header: true,
        complete: (results) => {
          resolve(this.parseCsv(results["data"]));
        },
      });
    });
  }

  private async parseCsv(data: object[]) {
    return new Promise<number[][]>((resolve) => {
      const res = data.map((item) => {
        return Object.values(item).map((value) => parseFloat(value));
      });
      resolve(res);
    });
  }

  private async shuffle(data: number[][], target: number[][]) {
    let counter = data.length;
    while (counter--) {
      const index = (Math.random() * counter) | 0;

      const tempData = data[counter];
      data[counter] = data[index];
      data[index] = tempData;

      const tempTarget = target[counter];
      target[counter] = target[index];
      target[index] = tempTarget;
    }
  }
}
