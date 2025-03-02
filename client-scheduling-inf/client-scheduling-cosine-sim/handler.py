from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import gc
import sys
import time
import subprocess
import os
import copy
from urllib.parse import parse_qs

sys.path.insert(0, "/home/app/function/utils")

from minio_data import readAndCreateModels


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


def apply_kmeans_on_parameters(model_parameters, n_clusters=2):
    """
    Apply K-Means clustering on the flattened parameters or gradients of models.

    Args:
    - model_parameters: A list of PyTorch tensors representing model parameters or gradients.
    - n_clusters: Number of clusters to form.

    Returns:
    - labels: The labels of each input vector after clustering.
    """
    # Flatten parameters or gradients
    flat_params = [
        flatten_parameters(params).cpu().detach().numpy() for params in model_parameters
    ]
    # Stack for KMeans
    data_matrix = np.stack(flat_params)
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_matrix)
    return kmeans.labels_


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


def calculate_cosine_similarity(model1_params, model2_params):
    """
    Calculate the cosine similarity between two sets of model parameters or gradients.

    Args:
    - model1_params: Flattened parameters or gradients of the first model.
    - model2_params: Flattened parameters or gradients of the second model.

    Returns:
    - similarity: Cosine similarity between the two sets of parameters or gradients.
    """
    # Flatten parameters or gradients
    flat_params1 = flatten_parameters(model1_params)
    flat_params2 = flatten_parameters(model2_params)
    # Calculate cosine similarity
    similarity = cosine_similarity(flat_params1.unsqueeze(0), flat_params2.unsqueeze(0))
    return similarity.item()


# Functions `flatten_parameters`, `apply_kmeans_on_parameters`, and `calculate_cosine_similarity` are defined as above


def main(req="round=1"):
    configs = parse_qs(req)
    round_no = int(configs["round"][0])
    print(f"Processing round {round_no}")
    round_no = str(round_no)
    gc.collect()
    clear_cache()
    similarities = {}
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

    # Apply K-Means clustering
    # cluster_labels = apply_kmeans_on_parameters(model_parameters, n_clusters=3)
    # print("Cluster labels:", cluster_labels)
    aggregated_model = aggregate_models(models)
    # # Example: Calculate cosine similarity between the aggregated model and all models and get top-k most similar models
    for client_id, model in zip(client_ids, models):
        similarities[client_id] = calculate_cosine_similarity(
            list(aggregated_model.parameters()), list(model.parameters())
        )
    top_k = 5
    top_k_models = dict(
        sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    )
    end_processing_time = time.time()
    end_total_time = time.time()

    fetch_time = end_fetch_time - start_fetch_time
    processing_time = end_processing_time - start_processing_time
    total_time = end_total_time - start_total_time
    cost_per_hour = 0.11988

    print(f"Total data fetch time: {fetch_time}")
    print(f"Total processing time: {processing_time}")
    print(f"Total time: {total_time}")
    print(f"Total cost: {total_time*cost_per_hour/3600}")
    print(f"Total communication cost: {fetch_time*cost_per_hour/3600}")
    print(f"Total processing cost: {processing_time*cost_per_hour/3600}")
    print("top_k_models:", top_k_models)


if __name__ == "__main__":
    main()


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    print(parse_qs(req))
    main(req)
    return
