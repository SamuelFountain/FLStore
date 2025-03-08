import os
from minio import Minio
from minio.error import S3Error
import pickle
import json
import torch
import torch.nn as nn
import torchvision.models as models  # For torchvision models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import io
import timm  # For accessing a wide variety of models including Swin Transformers
import zipfile  # For creating and reading zip files

# Use your JSON mapping file.
clients_info_json_file = os.path.join(
    os.path.dirname(__file__),
    "resnet18_cfarclients-10-of-250_rounds-2000_epochs-3_batchSize-20.json",
)

# Choose your model name.
# MODEL_NAME = 'mobilenet_v3_small'
# MODEL_NAME = 'efficientnet_v2_s'
MODEL_NAME = "resnet18"
DATASET_NAME = "CIFAR10"
# MODEL_NAME = 'swin_v2_t'


def get_minio_client():
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
            out_channels, in_channels, kernel_height, kernel_width = params[
                "weight"
            ].shape
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_height, kernel_width),
                bias=("bias" in params),
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
            model.add_module(layer_name, bn)
    return model


def create_model_by_copy(model_name, layers_data):
    if model_name in [
        "resnet18",
        "resnet34",
        "resnet50",
        "densenet121",
        "vgg16",
        "swin_v2_t",
        "efficientnet_v2_s",
        "mobilenet_v3_small",
    ]:
        model_func = getattr(models, model_name, None)
        if model_func is None:
            print("Model not supported")
            return None
        model = model_func()
    elif model_name.startswith("swin_v2_t"):
        model = timm.create_model(model_name, pretrained=False)
    else:
        print("Model not supported")
        return None
    model_dict = model.state_dict()
    layers_data = {k: v for k, v in layers_data.items() if k in model_dict}
    model_dict.update(layers_data)
    model.load_state_dict(model_dict)
    return model


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def read_key_value(client, bucket_name, layer_loc):
    response = client.get_object(bucket_name, layer_loc)
    data = response.read()
    try:
        object_data = pickle.loads(data)
    except pickle.UnpicklingError as e:
        print(f"Error unpickling data for {layer_loc}: {e}")
        object_data = None
    return object_data


def readAndCreateModels(round):
    """
    Fetches raw model data from MinIO for the given round using the JSON mapping,
    and creates complete model instances using the global MODEL_NAME.
    Returns a dictionary in the format:
      { round_key: { client_id: model_instance, ... } }
    """
    mapping = read_json(clients_info_json_file)
    client = get_minio_client()
    bucket_name = 'fl-layers-v3'
    try:
        client_models = {}
        round_key = str(round)
        if round_key not in mapping:
            print(f"Round {round_key} not found in JSON mapping. Using last available round.")
            round_key = list(mapping.keys())[-1]
        print(f"Fetching models for round {round_key}")
        for client_id, layer_map in mapping[round_key].items():
            client_id_str = str(client_id)
            # Fetch the raw layer data for this client.
            client_model_data = {}
            for layer_name, layer_loc in layer_map.items():
                layer_name_str = str(layer_name)
                client_model_data[layer_name_str] = read_key_value(client, bucket_name, layer_loc)
            # Convert the raw layer dictionary into a model instance.
            model_instance = create_model_by_copy(MODEL_NAME, client_model_data)
            client_models[client_id_str] = model_instance
        return {round_key: client_models}
    except S3Error as err:
        print(f"Error accessing MinIO: {err}")


def download_cifar10(download_dir="cifar10_data"):
    transform = transforms.Compose([transforms.ToTensor()])
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
        image_serialized = pickle.dumps(image)
        label_serialized = pickle.dumps(label)
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
                raise
    return data


def fetch_dataset(bucket_name="test-data", dataset_name="CIFAR10", test=True):
    client = get_minio_client()
    dataset = "test" if test else "train"
    images = fetch_data_with_prefix(
        client, bucket_name, f"{dataset_name}/{dataset}/images/"
    )
    labels = fetch_data_with_prefix(
        client, bucket_name, f"{dataset_name}/{dataset}/labels/"
    )
    return images, labels


def store_dataset_in_minio(bucket_name="test-data", dataset_name="CIFAR10"):
    client = get_minio_client()
    train_set, test_set = download_cifar10()
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    store_dataset(client, bucket_name, f"{dataset_name}/train", train_set)
    store_dataset(client, bucket_name, f"{dataset_name}/test", test_set)


def save_data_locally_for_rounds(
    rounds, zip_filename=f"{DATASET_NAME}_{MODEL_NAME}.zip"
):
    """
    For each round (e.g. [1,2,3,4,5]), fetch raw model data using readAndCreateModels.
    NOTE: readAndCreateModels returns a dict like { round_key: client_models }.
    Here we “flatten” it so that the saved data for round r is simply the client_models.
    Also fetches the test dataset and saves the original JSON mapping.
    All data is saved in a zip file.
    """
    model_data_all = {}
    for r in rounds:
        print(f"Fetching and saving raw model data for round {r}")
        temp = readAndCreateModels(str(r))
        # temp is like { round_key: client_models }; extract the inner dict.
        model_data_all[str(r)] = temp[str(r)]
    test_data = fetch_dataset(
        bucket_name="test-data", dataset_name="CIFAR10", test=True
    )
    original_mapping = read_json(clients_info_json_file)
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("model_data.pkl", pickle.dumps(model_data_all))
        zipf.writestr("test_data.pkl", pickle.dumps(test_data))
        zipf.writestr("json_mapping.pkl", pickle.dumps(original_mapping))
    print(f"Data for rounds {rounds} successfully saved to {zip_filename}")


def load_local_data(zip_filename=f"{DATASET_NAME}_{MODEL_NAME}.zip"):
    with zipfile.ZipFile(zip_filename, "r") as zipf:
        model_data = pickle.loads(zipf.read("model_data.pkl"))
        test_data = pickle.loads(zipf.read("test_data.pkl"))
        json_mapping = pickle.loads(zipf.read("json_mapping.pkl"))
    print(f"Loaded local data and JSON mapping from {zip_filename}")
    return model_data, test_data, json_mapping


def populate_minio_from_local_rounds_pattern(
    zip_filename=f"{DATASET_NAME}_{MODEL_NAME}.zip",
    model_bucket="fl-layers-v3",
    test_bucket="test-data",
):
    """
    Loads local data (raw model data, test dataset, and the original JSON mapping)
    and uploads each layer to MinIO using the original key pattern from the JSON mapping.
    If an entry is missing, a fallback key of "{round}/{client}/{layer}.pkl" is used.
    """
    local_model_data, test_data, saved_mapping = load_local_data(zip_filename)
    client = get_minio_client()
    if not client.bucket_exists(model_bucket):
        client.make_bucket(model_bucket)

    for round_key, clients_data in local_model_data.items():
        round_key_str = str(round_key)
        for client_id, layer_dict in clients_data.items():
            client_id_str = str(client_id)
            for layer_name, raw_data in layer_dict.items():
                layer_name_str = str(layer_name)
                # Look up target key in the saved JSON mapping
                target_key = (
                    saved_mapping.get(round_key_str, {})
                    .get(client_id_str, {})
                    .get(layer_name_str)
                )
                # print(f"Uploading layer '{layer_name_str}' for client '{client_id_str}' in round '{round_key_str}'")
                if target_key is None:
                    target_key = (
                        f"{round_key_str}/{client_id_str}/{layer_name_str}.pkl"
                    )
                    print(
                        f"Target key not found in JSON mapping; using fallback key '{target_key}'"
                    )
                # else:
                # print(f"Target key from JSON mapping: {target_key}")
                data_bytes = pickle.dumps(raw_data)
                stream = io.BytesIO(data_bytes)
                client.put_object(
                    model_bucket, target_key, stream, len(data_bytes)
                )
                # print(f"Uploaded layer '{layer_name_str}' for client '{client_id_str}' in round '{round_key_str}' to key '{target_key}'")

    # Upload test dataset
    images, labels = test_data
    if not client.bucket_exists(test_bucket):
        client.make_bucket(test_bucket)
    dataset_name = "CIFAR10"
    for idx, image in enumerate(images):
        data_bytes = pickle.dumps(image)
        stream = io.BytesIO(data_bytes)
        key = f"{dataset_name}/test/images/{idx}"
        client.put_object(test_bucket, key, stream, len(data_bytes))
    print(
        f"Uploaded {len(images)} test images to '{test_bucket}/{dataset_name}/test/images/'"
    )
    for idx, label in enumerate(labels):
        data_bytes = pickle.dumps(label)
        stream = io.BytesIO(data_bytes)
        key = f"{dataset_name}/test/labels/{idx}"
        client.put_object(test_bucket, key, stream, len(data_bytes))
    print(
        f"Uploaded {len(labels)} test labels to '{test_bucket}/{dataset_name}/test/labels/'"
    )

    print(
        "Repopulation of local data into MinIO using the original key patterns is complete."
    )


# --- Main Execution ---

# if __name__ == "__main__":
#     rounds_to_save = [1, 2, 3, 4, 5]
#     zip_filename = f"{DATASET_NAME}_{MODEL_NAME}.zip"

#     if not os.path.exists(zip_filename):
#         print(
#             f"Local data file '{zip_filename}' does not exist. Saving data locally."
#         )
#         save_data_locally_for_rounds(rounds_to_save, zip_filename=zip_filename)
#     else:
#         print(
#             f"Local data file '{zip_filename}' already exists. Skipping saving."
#         )

#     loaded_model_data, loaded_test_data, saved_mapping = load_local_data(
#         zip_filename=zip_filename
#     )
#     print("Loaded rounds from local data:", list(loaded_model_data.keys()))
#     if loaded_test_data:
#         print("Loaded test data length (images):", len(loaded_test_data[0]))

#     populate_minio_from_local_rounds_pattern(
#         zip_filename=zip_filename,
#         model_bucket="fl-layers-v3",
#         test_bucket="test-data",
#     )
