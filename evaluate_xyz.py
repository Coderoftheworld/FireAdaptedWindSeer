import random
import numpy as np
import torch

import windseer.data as data
import windseer.utils as utils
from windseer.nn.predict_model import get_prediction


MODEL_DIR = "/home/phantom/Documents/WindSeer/WindSeer/firebench_B_3to60_output/firebench_B_3to60"
DATASET = "/home/phantom/Documents/WindSeer/WindSeer/firebench 6m dataset/firebench_val.hdf5"

CHECKPOINTS = [f"e{i}" for i in range(1, 21)]
N_REPEATS = 1

device = torch.device("cpu")

for checkpoint in CHECKPOINTS:
    print("\n" + "=" * 60)
    print(f"EVALUATING {checkpoint}")
    print("=" * 60)

    net, params = utils.load_model(
        MODEL_DIR, checkpoint, None, device, True
    )

    dataset_kwargs = params.Dataset_kwargs()

    testset = data.HDF5Dataset(
        DATASET,
        **dataset_kwargs
    )

    channels = ["ux", "uy", "uz"]

    stats = {
        ch: {
            "abs": 0.0,
            "sq": 0.0,
            "err": 0.0,
            "abs_true": 0.0,
            "n": 0,
            "sample_mae": []
        }
        for ch in channels
    }

    with torch.no_grad():

        for repeat in range(N_REPEATS):

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

                outputs = prediction["pred"]

                terrain = inputs[0, 0]
                air_mask = terrain > 0

                for ch in channels:

                    index = params.data["label_channels"].index(ch)

                    pred = outputs[0, index]
                    true = labels[0, index]

                    error = pred[air_mask] - true[air_mask]

                    s = stats[ch]

                    s["abs"] += error.abs().sum().item()
                    s["sq"] += (error ** 2).sum().item()
                    s["err"] += error.sum().item()
                    s["abs_true"] += true[air_mask].abs().sum().item()
                    s["n"] += error.numel()

                    s["sample_mae"].append(
                        error.abs().mean().item()
                    )

    print()
    print("========================================")
    print("3-D wind validation results")
    print("========================================")
    print(f"Validation CFD flows: {len(testset)}")
    print(f"Repeats per flow:     {N_REPEATS}")
    print(f"Total predictions:    {len(testset) * N_REPEATS}")
    print()

    for ch in channels:

        s = stats[ch]

        mae = s["abs"] / s["n"]
        rmse = (s["sq"] / s["n"]) ** 0.5
        bias = s["err"] / s["n"]
        mean_abs_true = s["abs_true"] / s["n"]

        print(ch.upper())
        print(f"  MAE:              {mae:.4f} m/s")
        print(f"  RMSE:             {rmse:.4f} m/s")
        print(f"  Bias:             {bias:+.4f} m/s")
        print(f"  Mean |true|:      {mean_abs_true:.4f} m/s")
        print(f"  Mean sample MAE:  {np.mean(s['sample_mae']):.4f} m/s")
        print(f"  Std sample MAE:   {np.std(s['sample_mae']):.4f} m/s")
        print()

    print("========================================")
