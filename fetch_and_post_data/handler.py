from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import gc
import sys
import time
import subprocess
import random
import os
from urllib.parse import parse_qs


sys.path.insert(0, "/home/app/function/utils")

from minio_data_local_save import readAndCreateModels, get_minio_client
from minio_data_local_save import fetch_dataset


def clear_cache():
    try:
        # Requires root privileges
        subprocess.run(["sync"])  # Flush file system buffers
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
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


def model_inferece(dataset, model):
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    testLoader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2
    )
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def fetch_data_from_minio(
    bucket_name="populated-test-data", dataset_name="CIFAR10", test=True, sample_propotion=1.0
):
    """args:
    bucket_name: Name of the bucket
    dataset_name: Name of the dataset
    test: If True, fetch the test dataset, else fetch the training dataset
    returns:
    dataset: List of tuples of images and labels
    """
    images, labels = fetch_dataset(
        bucket_name, dataset_name, test
    )  # Assuming this fetches images and labels correctly
    dataset = list(zip(images, labels))  # Pair each image with its corresponding label
    # randomly sample the dataset
    if sample_propotion < 1:
        dataset = random.sample(dataset, int(len(dataset) * sample_propotion))

    return dataset


def main():
    PER_CLIENT = False
    for round_no in range(1, 2):
        accuracies = []
        print(f"Processing round {round_no}")
        round_no = str(round_no)
        # gc.collect()
        # clear_cache()
        start_total_time = time.time()
        start_total_time = time.time()
        round_no = str(round_no)
        start_fetch_time = time.time()
        client_models_per_round = readAndCreateModels(round_no)
        dataset = fetch_data_from_minio()
        end_fetch_time = time.time()
        start_processing_time = time.time()
        client2models = client_models_per_round[round_no]
        client_ids = list(client2models.keys())
        models = list(client2models.values())

        if PER_CLIENT:
            for model in models:
                # Model inference
                accuracies.append(model_inferece(dataset, model))
        else:
            # Model inference
            accuracies.append(model_inferece(dataset, models[0]))

        # Apply inference on the models
        end_processing_time = time.time()
        end_total_time = time.time()

        fetch_time = end_fetch_time - start_fetch_time
        processing_time = end_processing_time - start_processing_time
        total_time = end_total_time - start_total_time
        cost_per_hour = 0.922

        print(f"Accuracies: {accuracies}")
        print(f"Total data fetch time: {fetch_time}")
        print(f"Total processing time: {processing_time}")
        print(f"Total time: {total_time}")
        print(f"Total cost: {total_time*cost_per_hour/3600}")
        print(f"Total communication cost: {fetch_time*cost_per_hour/3600}")
        print(f"Total processing cost: {processing_time*cost_per_hour/3600}")
        return str(accuracies)


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    print(f"req is {repr(req)}: Start")
    req = main()
    print(f"req is {repr(req)}: End")
    return req
