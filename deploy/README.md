# Azure Kubernetes Deployment Manual

## Pre-requisite

`azure-cli, kubectl`

## Cluster Deployment

```bash
# 0. set account subscription
$ az account set --subscription 7c378aaf-2dce-4a2c-af50-d76c13bbc811

# 1. create resource group

$ az group create --name secureml --location <location>

## locations support confidential computing: canadacentral, canadaeast, eastus, southcentralus, westus2

# 2. enable gen2vm&startstop feature and register the features

$ az extension add --name aks-preview
$ az feature register --name Gen2VMPreview --namespace Microsoft.ContainerService
$ az feature register --name StartStopPreview --namespace Microsoft.ContainerService
$ az provider register -n Microsoft.ContainerService

# 3. create the cluster

$ az aks create --name <cluster name>\
    --resource-group secureml \
    --node-vm-size Standard_DC8_v2 \
    --node-count <node count> \
    --enable-addon confcom \
    --network-plugin azure \
    --vm-set-type VirtualMachineScaleSets \
    --aks-custom-headers usegen2vm=true

# 4. get credentials, to use kubectl

$ az aks get-credentials --resource-group secureml --name <cluster name>

# 5. stop/delete clusters

$ az aks stop/delete --resource-group secureml --name <cluster name> 

```

## Test Script

1. edit ./templates/run.sh, fill in the blanks

`${GIT_USERNAME}`: username of your github account

`${GIT_PASSWORD}`: password of your github account

`${COMMIT_HASH}`: the hash of the project commit, which you're currently using for experiment

`${STORAGE_NODE_HOSTNAME}`: the name of node to deploy storage service

`${AGGREGATE_NODE_HOSTNAME}`: the name of node to deploy aggregation service

`${FRONTEND_NODE_HOSTNAME}`: the name of node to deploy frontend

`${MONGODB_IP}`: the IP address of node with MongoDB

2. tag training nodes
```bash
# label all the nodes for worker deployment with command:
$ kubectl label <node-hostname> secureml_app=training
```


3. start storage service and upload

```bash
# create aks-ssh pod and enter storage node, start mongoDB

# start the storage service
$ ./run.sh upload_retina start
```

4. deploy the workload over the cluster

```bash

# start the workload
$ ./run.sh all start

# stop the workload
$ ./run.sh all stop

```