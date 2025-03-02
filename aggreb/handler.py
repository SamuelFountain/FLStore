from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import gc
import sys
import time
import subprocess
import os
from urllib.parse import parse_qs


sys.path.insert(0, "/home/app/function/utils")
sys.path.insert(0, "./utils")

from minio_data import readAndCreateModels


def clear_cache():
    try:
        # Requires root privileges
        subprocess.run(["sync"])  # Flush file system buffers
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
        )
        print("Cache cleared successfully.")
        # Clear swap
        subprocess.run(["sudo", "swapoff", "-a"], check=True)
        subprocess.run(["sudo", "swapon", "-a"], check=True)
        print("Swap memory cleared.")
    except Exception as e:
        print(f"Failed to clear cache: {e}")


def flatten_parameters(params):
    """
    Flatten parameters or gradients of a model into a single vector.
    """
    vec = []
    for param in params:
        vec.append(param.view(-1))
    return torch.cat(vec)


def main(req):
    round_no = str(req)
    # for round_no in range(1, 3):
    print(f"Processing round {round_no}")
    round_no = str(round_no)
    gc.collect()
    clear_cache()
    print("starting load")
    start_total_time = time.time()
    start_total_time = time.time()
    round_no = str(round_no)
    start_fetch_time = time.time()
    client_models_per_round = readAndCreateModels(round_no)
    end_fetch_time = time.time()
    start_processing_time = time.time()
    client2models = client_models_per_round[round_no]
    client_ids = list(client2models.keys())
    models = list(client2models.values())
    # Example usage for K-Means clustering
    # Assume `models` is a list of PyTorch models
    model_parameters = [
        list(model.parameters()) for model in models
    ]  # Collect parameters from each model
    print("-----")
    summed = [
        model_parameters[0][x].data.clone()
        for x in range(len(model_parameters[0]))
    ]

    # Sum all other parameters
    for p in model_parameters[1:]:
        for i in range(len(summed)):
            summed[i] += p[i].data

    # Compute the average
    for i in range(len(summed)):
        summed[i] = summed[i] / len(model_parameters)

    end_processing_time = time.time()
    end_total_time = end_processing_time
    fetch_time = end_fetch_time - start_fetch_time
    processing_time = end_processing_time - start_processing_time
    total_time = end_total_time - start_total_time
    cost_per_hour = 0.922

    print(f"Total data fetch time: {fetch_time}")
    print(f"Total processing time: {processing_time}")
    print(f"Total time: {total_time}")
    print(f"Total cost: {total_time*cost_per_hour/3600}")
    print(f"Total communication cost: {fetch_time*cost_per_hour/3600}")
    print(
        f"Total processing cost: {processing_time*cost_per_hour/3600}",
        flush=True,
    )


if __name__ == "__main__":
    print("Hi")
    # main()


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    print("start")
    req = main(req.split("=")[1])
    print("end")
    return req
