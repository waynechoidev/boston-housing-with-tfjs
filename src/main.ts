import { BostonHousingDataset } from "./dataset";
import "./style.css";

// Dataset
const data = new BostonHousingDataset();

document.getElementById("load-button")?.addEventListener("click", async () => {
  await data.loadData();
  console.log(data.dataset);
});
