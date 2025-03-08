#!/usr/bin/bash

# Installing Software
## TODO DOCKER
## TODO CONDA
source set_conda.sh
## kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
## arkade (for installing openfaas)
curl -sLS https://get.arkade.dev | sudo sh

## faas-cli

curl -SLsf https://cli.openfaas.com | sudo sh

## k3sup for setting up cluster for openfaas

curl -sLS https://get.k3sup.dev | sh
sudo install k3sup /usr/local/bin/

# Setting up MinIO
if [ ! -d "data" ]; then
  mkdir data
  echo "Directory 'data' created."
else
  echo "Directory 'data' already exists."
fi
docker run -p 9000:9000 -d -p 9001:9001 -e "MINIO_ROOT_USER=minio99" -e "MINIO_ROOT_PASSWORD=minio123" quay.io/minio/minio server ./data --console-address ":9001"

# Set up data 
if [ ! -f "./fetch_and_post_data/CIFAR10_resnet18.zip" ]; then
  wget --header="Authorization: Bearer eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0ZDNkYjA4LTQ5ZDItNGI2Ny1hYWUzLTU3OGMzMzc1ZWFjOCIsImRhdGEiOnt9LCJyYW5kb20iOiIzYmRiOWFmNWJlYjI2MmM4OTViMDIyNzE5YjdlZDdjZiJ9.L-VIkeBpUY4SyfVnVyPndiNcvGsbZfqkHvvs9nZz6-n-Myre7awFNi3TeCYRaxAYjQGHylR6-G_2MpITJhH4Ew" \
  "https://zenodo.org/record/14986611/files/CIFAR10_resnet18.zip?download=1" -O ./fetch_and_post_data/CIFAR10_resnet18.zip
else
  echo "Local data already present"
fi

cd ./fetch_and_post_data
python3 minio_data.py
cd .. 
# Setting Up Cluster


# Define the k3s version
K3S_VERSION="v1.24"


# Install the master node




# k3sup install --local --k3s-channel $K3S_VERSION \
#     --k3s-extra-args '--write-kubeconfig-mode 644 --write-kubeconfig ~/.kube/config --disable traefik --disable metrics-server --disable local-storage --disable servicelb' \
#     --local-path /tmp/config




# echo "Waiting for all pods to become Ready..."
# export KUBECONFIG=/tmp/config
# kubectl config use-context default
# # Wait until at least one pod exists in kube-system
# until kubectl get pods -n kube-system | grep -q 'core'; do
#     echo "Waiting for pods to be created..."
#     sleep 2
# done
# echo "Found core dns, still waiting for pods to be created..."
# # Now wait for all pods to be Ready
# kubectl wait --for=condition=Ready pods --all --timeout=600s -n kube-system

# kubectl get node -o wide
# echo "All pods are Ready."

# echo "k3s cluster installation complete."

# # Setting up open Faas
# export TIMEOUT=11m
# export HARD_TIMEOUT=12m2s

# arkade install openfaas --set=faasIdler.dryRun=false \
#   --set gateway.upstreamTimeout=$TIMEOUT \
#   --set gateway.writeTimeout=$HARD_TIMEOUT \
#   --set gateway.readTimeout=$HARD_TIMEOUT


# until kubectl get pods -n openfaas | grep -q 'nats'; do
#     echo "Waiting for pods to be created..."
#     sleep 2
# done
# echo "Found nats, still waiting for pods to be created..."
# kubectl wait --for=condition=available --timeout=600s deployment --all --namespace openfaas
# kubectl get deploy --namespace openfaas

# kubectl rollout status -n openfaas deploy/gateway

# kubectl port-forward -n openfaas svc/gateway 8080:8080 &

# sleep 1
# export PASSWORD=$(
#   kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode
#   echo
#       )
# echo -n $PASSWORD | faas-cli login --username admin --password-stdin

# sleep 1
cd functions

# faas-cli deploy
# faas-cli list
faas-cli local-run infr --network host