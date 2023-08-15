import { BostonHousingDataset } from "./dataset";
import "./style.css";
import { Tensors } from "./tensors";

// Dataset
const data = new BostonHousingDataset();
const tensors = new Tensors();

document.getElementById("load-button")?.addEventListener("click", async () => {
  await data.loadData();
  tensors.init(data.dataset);
  console.log(tensors.testFeatures);

  const baseline = tensors.baseline;
  console.log(baseline);
  document.getElementById(
    "base-loss"
  )!.innerHTML = `Baseline Loss (Mean Squared Error): ${baseline.toFixed(2)}`;
});
