import os
from minio import Minio
from minio.error import S3Error
import pickle
import json
import torch
import torch.nn as nn
import torchvision.models as models  # Added import for torchvision models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import OrderedDict
import time
import io
import timm  # Import TIMM for accessing a wide variety of models including Swin Transformers

module_dir = os.path.dirname(__file__)
# clients_info_json_file = os.path.join(
#     module_dir,
#     "json_for_Ahmad/efficientnet_v2_smallclients-10-of-250_rounds-2000_epochs-3_batchSize-20.json",
# )

clients_info_json_file = (
    "resnet18_cfarclients-10-of-250_rounds-2000_epochs-3_batchSize-20.json"
)


def get_minio_client():
    # Create a MinIO client with the provided credentials
    client = Minio(
        "127.0.0.1:9000",
        access_key="minio99",
        secret_key="minio123",
        secure=False,  # Set to True if MinIO server supports HTTPS
    )
    return client


def create_model(layers_data):
    model = nn.Sequential()

    for layer_name, params in layers_data.items():
        if "conv" in layer_name and "weight" in params:
            # Assuming params['weight'] is a 4D tensor for convolutional layers
            out_channels, in_channels, kernel_height, kernel_width = params[
                "weight"
            ].shape
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_height, kernel_width),
                bias="bias" in params,
            )
            conv.weight.data = params["weight"]
            if "bias" in params:
                conv.bias.data = params["bias"]
            model.add_module(layer_name, conv)
        elif "bn" in layer_name:
            num_features = params["weight"].shape[0]
            bn = nn.BatchNorm2d(num_features)
            bn.weight.data = params["weight"]
            bn.bias.data = params["bias"]
            bn.running_mean.data = params["running_mean"]
            bn.running_var.data = params["running_var"]
            # Note: num_batches_tracked is a scalar and doesn't need to be directly assigned for model functionality
            model.add_module(layer_name, bn)
        # Extend this with other layer types as needed

    return model


def create_model_by_copy(model_name, layers_data):
    # Extended model selection to include models from TIMM
    if model_name in [
        "resnet18",
        "resnet34",
        "resnet50",
        "densenet121",
        "vgg16",
        "swin_v2_t",
        "efficientnet_v2_s",
    ]:
        model_func = getattr(models, model_name, None)
        if model_func is None:
            print("Model not supported")
            return None
        model = model_func()
    elif model_name.startswith(
        "swin"
    ):  # Check if the model is a Swin V2 model
        model = timm.create_model(
            model_name, pretrained=False
        )  # Create model without pretrained weights
    else:
        print("Model not supported")
        return None
    # print(f"Model: {model}")
    model_dict = model.state_dict()
    # Proceed with filtering, updating, and loading the state dict
    layers_data = {k: v for k, v in layers_data.items() if k in model_dict}
    model_dict.update(layers_data)
    model.load_state_dict(model_dict)
    return model


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def read_key_value(client, bucket_name, layer_loc):
    # Get the HTTPResponse object
    response = client.get_object(bucket_name, layer_loc)
    # Read the content of the file from the response
    data = response.read()
    # If your data is pickled, unpickle it; otherwise, adjust according to your data format
    try:
        # Attempt to unpickle the data if it's expected to be pickled
        object_data = pickle.loads(data)
    except pickle.UnpicklingError as e:
        # Handle or log errors if the data is not in the expected format
        print(f"Error unpickling data for {layer_loc}: {e}")
        object_data = None  # or handle as appropriate
    # Store the actual data instead of the response object
    return object_data


def readAndCreateModels(round):
    clients_info_dict = read_json(clients_info_json_file)

    # Create a MinIO client with the provided credentials
    client = Minio(
        "127.0.0.1:9000",
        access_key="minio99",
        secret_key="minio123",
        secure=False,  # Set to True if MinIO server supports HTTPS
    )

    # Specify the bucket name
    bucket_name = "fl-layers-v3"

    try:
        # List objects in the specified bucket
        objects = client.list_objects(bucket_name, recursive=True)
        # print(f"Objects in '{bucket_name}' bucket:")

        # rounds = clients_info_dict.keys()
        client_models_per_round = {}
        # for r in rounds:
        if round not in clients_info_dict.keys():
            print(
                "Round not found in the clients info file, sending random round results"
            )
            round = list(clients_info_dict.keys())[-1]
        # print(f"Fetching models for round {round}")
        clients = clients_info_dict[round].keys()
        client_models = {}
        read_times = []
        model_creation_times = []
        for c in clients:
            client_models[c] = {}
            client_model = {}
            start_read_time = time.time()
            for layer_name, layer_loc in clients_info_dict[round][c].items():
                # print(f"Layer name: {layer_name} for client {c}")
                object_data = read_key_value(client, bucket_name, layer_loc)
                # Store the actual data instead of the response object
                client_model[layer_name] = object_data

                # print(f"Object {layer_name} for client {c} in round {r}: {client_model[layer_name]}")
            # Create a model based on the layers data
            # client_models[c] = create_model(client_model)
            end_read_time = time.time()
            read_times.append(end_read_time - start_read_time)
            start_model_create_time = time.time()
            client_models[c] = create_model_by_copy("resnet18", client_model)
            end_model_create_time = time.time()
            model_creation_times.append(
                end_model_create_time - start_model_create_time
            )
            # print('model created for client', c)
            # client_models[c] = client_model
        # print('Average read time per model: ', sum(read_times) / len(read_times))
        # print('Average model creation time: ', sum(model_creation_times) / len(model_creation_times))
        # print('Total models read time: ', sum(read_times))
        # print('Total models creation time: ', sum(model_creation_times))
        client_models_per_round[round] = client_models

        # print('Models created for clients: ', client_models_per_round[round].keys())
        return client_models_per_round
    except S3Error as err:
        print(f"Error accessing MinIO: {err}")


def download_cifar10(download_dir="cifar10_data"):
    # Transform the PIL Images to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download the training and test set
    train_set = datasets.CIFAR10(
        root=download_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=download_dir, train=False, download=True, transform=transform
    )

    return train_set, test_set


def store_dataset(client, bucket_name, dataset_name, dataset):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    for idx, (image, label) in enumerate(dataset):
        # Serialize and store each image and label
        image_serialized = pickle.dumps(image)
        label_serialized = pickle.dumps(label)

        # Convert bytes data to a file-like object
        image_stream = io.BytesIO(image_serialized)
        label_stream = io.BytesIO(label_serialized)

        client.put_object(
            bucket_name,
            f"{dataset_name}/images/{idx}",
            image_stream,
            len(image_serialized),
        )
        client.put_object(
            bucket_name,
            f"{dataset_name}/labels/{idx}",
            label_stream,
            len(label_serialized),
        )


def fetch_data_with_prefix(client, bucket_name, prefix):
    """args:
    client: MinIO client
    bucket_name: Name of the bucket
    prefix: Prefix of the object names to fetch
    returns:
    data: List of objects fetched from MinIO
    """
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    data = []

    for obj in objects:
        try:
            object_data = client.get_object(bucket_name, obj.object_name)
            data.append(pickle.loads(object_data.read()))
        except S3Error as e:
            if e.code == "NoSuchKey":
                print(f"Object {obj.object_name} does not exist.")
            else:
                raise  # Re-raise the exception if it's not a NoSuchKey error
    return data


def fetch_dataset(bucket_name="test-data", dataset_name="CIFAR10", test=True):
    """Fetches the dataset from MinIO and returns the images and labels"""
    client = get_minio_client()
    dataset = "test" if test else "train"
    # images = ""
    # labels = ""
    images = fetch_data_with_prefix(
        client, bucket_name, f"{dataset_name}/{dataset}/images/"
    )
    labels = fetch_data_with_prefix(
        client, bucket_name, f"{dataset_name}/{dataset}/labels/"
    )
    # print(f"Fetched {len(images)} images.")
    # print(f"images shape: {images[0].shape}")
    return images, labels


def store_dataset_in_minio(bucket_name="test-data", dataset_name="CIFAR10"):
    """Stores the dataset in MinIO"""
    client = get_minio_client()
    # Example usage
    train_set, test_set = download_cifar10()

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    # Store the train and test sets in MinIO under "CIFAR10" dataset name
    store_dataset(client, bucket_name, f"{dataset_name}/train", train_set)
    store_dataset(client, bucket_name, f"{dataset_name}/test", test_set)
