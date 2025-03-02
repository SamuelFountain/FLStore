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
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil 

sys.path.insert(0, "/home/app/function/utils")


from minio import Minio
from minio.error import S3Error

def clear_cache():
    try:
        gc.collect()
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


def get_minio_client():
    # Create a MinIO client with the provided credentials
    client = Minio(
        "192.168.0.15:9000",
        access_key="UhNp0B2T6EVSVWLWZJux",
        secret_key="jzBiNer8CG1bEXbjsiwBHUfWLfUdm39Dm2K5Aor6",
        secure=False,  # Set to True if MinIO server supports HTTPS
    )
    return client

def upload_files_to_minio(client, bucket_name, source_directory):
    """
    Upload all files from a local directory to a MinIO bucket.

    :param client: MinIO client
    :param bucket_name: Name of the MinIO bucket
    :param source_directory: Local directory containing files to upload
    """
    # Ensure the bucket exists
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        if os.path.isfile(file_path):
            client.fput_object(bucket_name, filename, file_path)
            print(f"Uploaded {filename} to bucket {bucket_name}.")
            
            
def download_files_from_minio(client, bucket_name, destination_directory):
    """
    Download all files from a MinIO bucket to a local directory.

    :param client: MinIO client
    :param bucket_name: Name of the MinIO bucket
    :param destination_directory: Local directory to save downloaded files
    """
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        local_file_path = os.path.join(destination_directory, obj.object_name)
        client.fget_object(bucket_name, obj.object_name, local_file_path)
        print(f"Downloaded {obj.object_name} to {local_file_path}.")


def main(incoming):
    clear_cache()
    print("incoming is ", incoming)
    client = get_minio_client()
    start_fetch_time = time.time()
    

    source = "./llama_3_2_1B_weights"
    os.mkdir(source)
    download_files_from_minio(client, "llama-3-2-1b-weights", source)    
    end_fetch_time = time.time()
    
    
    start_processing_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    end_processing_time = time.time()
    end_total_time = time.time()

    fetch_time = end_fetch_time - start_fetch_time
    start_total_time =  start_fetch_time
    processing_time = end_processing_time - start_processing_time
    total_time = end_total_time - start_total_time
    cost_per_hour = 0.922
    input_text = f"You are writer #{incoming}. Once upon a time, in a faraway land, there was a"

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output using the model
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

    # Decode the output to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    accuracies = [output_text]
    print(f"Accuracies: {accuracies}")
    print(f"Total data fetch time: {fetch_time}")
    print(f"Total processing time: {processing_time}")
    print(f"Total time: {total_time}")
    print(f"Total cost: {total_time*cost_per_hour/3600}")
    print(f"Total communication cost: {fetch_time*cost_per_hour/3600}")
    print(f"Total processing cost: {processing_time*cost_per_hour/3600}")
    shutil.rmtree(source)
    return str(accuracies)


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    print("Hi")
    print(f"req is {repr(req)}: Start")
    req = main(req.split("=")[1])
    print(f"req is {repr(req)}: End")
    return req
