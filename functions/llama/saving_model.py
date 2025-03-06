from transformers import AutoModel, AutoTokenizer
import os
from minio import Minio
from minio.error import S3Error
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the saved model and tokenizer
save_directory = "./llama_3_2_1B_weights"
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Example input
#input_text = "Once upon a time, in a faraway land, there was a"

# Tokenize input text
#input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate output using the model
# output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

# Decode the output to text
#output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text:")
#print(output_text)




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
    print("check")
    if not client.bucket_exists(bucket_name):
        print("Trying to make")
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
        
print("getting Mode")
# Load the LLaMA model (assuming it's hosted on Hugging Face or you've downloaded it locally)
model_name = "meta-llama/Llama-3.2-1B"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name)

# Optionally, load the tokenizer if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Saving model")
# Save the model and tokenizer to a directory
save_directory = "./llama_3_2_1B_weights"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)        

print("moving to minio")
client = get_minio_client()
upload_files_to_minio(client, "llama-3-2-1b-weights", "./llama_3_2_1B_weights")

""" print("getting from mino")
source = "./pulling"
oas.mkdir(source)
download_files_from_minio(client, "testing-dir-bucket", source) """