import { ModelType } from "./constant";
import { BostonHousingDataset } from "./dataset";
import "./style.css";
import { Tensors } from "./tensors";

// Dataset
const data = new BostonHousingDataset();
const tensors = new Tensors({
  numEpochs: 200,
  batchSize: 40,
  learningRate: 0.01,
});

document.getElementById("load-button")?.addEventListener("click", async () => {
  await data.loadData();
  tensors.init(data.dataset);

  const baseline = tensors.baseline;
  document.getElementById(
    "base-loss"
  )!.innerHTML = `Baseline Loss (Mean Squared Error): ${baseline.toFixed(2)}`;
  (document.getElementById("model-dropdown") as HTMLSelectElement).disabled =
    false;
});

const modelDropdown = document.getElementById(
  "model-dropdown"
) as HTMLSelectElement;
modelDropdown.addEventListener("change", () => {
  const modelType = parseInt(modelDropdown.value) as ModelType;
  tensors.modelType = modelType;
  document.getElementById("model-label")!.innerHTML = tensors.modelLabel;
  (document.getElementById(`train-button`) as HTMLButtonElement).disabled =
    false;
  tensors.cleanResult();
});

document
  .getElementById("train-button")
  ?.addEventListener("click", () => tensors.trainModel());
