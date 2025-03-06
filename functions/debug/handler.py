import subprocess
import logging
import time
from dotmap import DotMap
from pytorch_lightning import seed_everything
from torch.nn.init import kaiming_uniform_
import sys
from urllib.parse import parse_qs


sys.path.insert(0, "/home/app/function/utils")

from minio_data import readAndCreateModels, get_minio_client


# ====== Simulation Config ======
args = DotMap()
args.lr = 0.001
args.weight_decay = 0.0001
args.batch_size = 512

args.model = "resnet18"  # [resnet18, resnet34, resnet50, densenet121, vgg16]
args.epochs = 25  # range 10-25
args.dataset = "cifar10"  # ['cifar10', 'femnist']
args.clients = 5  # keep under 30 clients and use Resnet18, Resnet34, or Densenet to evaluate on Colab
args.faulty_clients_ids = "0"  # can be multiple clients separated by comma e.g. "0,1,2"  but keep under args.clients clients and at max less than 7
args.noise_rate = 1  # noise rate 0 to 1
args.sampling = "iid"  # [iid, "niid"]


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    script_path = "/home/app/function/FedDebug/fault-localization/artifact.py"
    subprocess.run(["python3", script_path])
    return req
