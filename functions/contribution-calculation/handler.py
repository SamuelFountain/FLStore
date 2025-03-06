from sklearn.cluster import KMeans
import numpy as np
import copy
import torch
from torch.nn.functional import cosine_similarity
import gc
import sys
import time
import subprocess
import os
import random
from urllib.parse import parse_qs


sys.path.insert(0, "/home/app/function/utils")

from minio_data import readAndCreateModels
from minio_data import fetch_dataset


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


def model_inference(dataset, model):
    """

    Perform inference with a single model and calculate the accuracy.



    Args:

    dataset (list): The dataset used for evaluation.

    model (PyTorch model): The model to evaluate.



    Returns:

    float: The accuracy of the model.

    """

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


def aggregate_models(models):
    """

    Aggregate multiple models by averaging their parameters.



    Args:

    models (list of PyTorch models): Models to be aggregated.



    Returns:

    PyTorch model: The aggregated model.

    """

    aggregated_model = copy.deepcopy(models[0])

    param_keys = list(aggregated_model.state_dict().keys())

    # Sum parameters from all models

    for key in param_keys:
        summed_params = sum([model.state_dict()[key] for model in models])

        averaged_params = summed_params / len(models)

        aggregated_model.state_dict()[key].copy_(averaged_params)

    return aggregated_model


def model_performance(dataset, model):
    """

    Calculate the performance of a model on a given dataset.



    Args:

    dataset (list): The dataset used for evaluation.

    model (PyTorch model): The model to evaluate.



    Returns:

    float: The accuracy of the model.

    """

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


def calculate_shapley_values(client_ids, models, dataset):
    """

    Calculate the Shapley values by evaluating the performance impact of leaving out each model.



    Args:

    client_ids (list): List of client IDs.

    models (list): List of PyTorch models corresponding to client IDs.

    dataset (list): The dataset used for evaluation.



    Returns:

    dict: A dictionary of client IDs to their Shapley values.

    """

    all_models = models

    full_model = aggregate_models(all_models)

    full_performance = model_performance(dataset, full_model)

    shapley_values = {}

    for client_id, model in zip(client_ids, models):
        # Aggregate all models except the current one

        other_models = [m for m in all_models if m != model]

        if other_models:
            aggregated_other_models = aggregate_models(other_models)

            performance_without_current = model_performance(
                dataset, aggregated_other_models
            )

        else:
            performance_without_current = (
                0  # If no other models, the performance is assumed zero
            )

        # Calculate the marginal contribution

        marginal_contribution = full_performance - performance_without_current

        shapley_values[client_id] = marginal_contribution

    return shapley_values


def fetch_data_from_minio(
    bucket_name="test-data",
    dataset_name="CIFAR10",
    test=True,
    sample_propotion=1.0,
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

    dataset = list(
        zip(images, labels)
    )  # Pair each image with its corresponding label

    # randomly sample the dataset

    if sample_propotion < 1:
        dataset = random.sample(dataset, int(len(dataset) * sample_propotion))

    return dataset


def main():
    args = sys.argv

    ROUNDS = 500

    PER_CLIENT = False

    for round_no in range(1, 50):
        accuracies = []

        print(f"Processing round {round_no}")

        round_no = str(round_no)

        gc.collect()

        clear_cache()

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

        shapley_vals = calculate_shapley_values(client_ids, models, dataset)

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

        print("Shapley Values:", shapley_vals)


if __name__ == "__main__":
    main()


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    main()
    return req
