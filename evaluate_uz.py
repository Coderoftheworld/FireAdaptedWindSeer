import random
import numpy as np
import torch

import windseer.data as data
import windseer.utils as utils
from windseer.nn.predict_model import get_prediction


MODEL_DIR = "cpu_sparse_5min_output/cpu_sparse_5min"
DATASET = "data/medium/validation_20.hdf5"

# Evaluate each CFD flow with several different deterministic
# crops / synthetic UAV trajectories.
N_REPEATS = 5

device = torch.device("cpu")

# Load architecture, config, and trained weights from latest.model
net, params = utils.load_model(
    MODEL_DIR,
    "latest",
    None,
    device,
    True
)

# Construct validation dataset using exactly the parameters
# stored with the trained model.
dataset_kwargs = params.Dataset_kwargs()
testset = data.HDF5Dataset(
    DATASET,
    **dataset_kwargs
)

uz_index = params.data["label_channels"].index("uz")

sum_abs_error = 0.0
sum_squared_error = 0.0
sum_error = 0.0
sum_abs_true_uz = 0.0
n_cells = 0

per_run_mae = []

with torch.no_grad():
    for repeat in range(N_REPEATS):

        # Fixed seeds make the random crops and UAV trajectories
        # reproducible if this script is run again.
        seed = 12345 + repeat
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for i in range(len(testset)):
            sample = testset[i]

            input_tensor = sample[0]
            label_tensor = sample[1]

            scale = 1.0
            if params.data["autoscale"]:
                scale = sample[3].item()

            prediction, inputs, labels = get_prediction(
                input_tensor,
                label_tensor,
                scale,
                device,
                net,
                params,
                False,
                False
            )

            pred = prediction["pred"]

            # get_prediction() adds a batch dimension
            pred_uz = pred[0, uz_index]
            true_uz = labels[0, uz_index]

            # Terrain is encoded as zero.
            # Positive terrain-distance values correspond to flow cells.
            terrain = inputs[0, 0]
            air_mask = terrain > 0

            error = pred_uz[air_mask] - true_uz[air_mask]

            if error.numel() == 0:
                continue

            sum_abs_error += error.abs().sum().item()
            sum_squared_error += (error ** 2).sum().item()
            sum_error += error.sum().item()
            sum_abs_true_uz += true_uz[air_mask].abs().sum().item()
            n_cells += error.numel()

            per_run_mae.append(error.abs().mean().item())


mae = sum_abs_error / n_cells
rmse = (sum_squared_error / n_cells) ** 0.5
bias = sum_error / n_cells
mean_abs_true = sum_abs_true_uz / n_cells

print()
print("========================================")
print("Vertical wind validation results")
print("========================================")
print(f"Validation CFD flows:     {len(testset)}")
print(f"Repeats per flow:         {N_REPEATS}")
print(f"Total predictions:        {len(testset) * N_REPEATS}")
print(f"Evaluated air cells:      {n_cells:,}")
print()
print(f"Uz MAE:                   {mae:.4f} m/s")
print(f"Uz RMSE:                  {rmse:.4f} m/s")
print(f"Uz bias:                  {bias:+.4f} m/s")
print(f"Mean |true Uz|:           {mean_abs_true:.4f} m/s")
print()
print(f"Mean sample MAE:          {np.mean(per_run_mae):.4f} m/s")
print(f"Std. sample MAE:          {np.std(per_run_mae):.4f} m/s")
print("========================================")
