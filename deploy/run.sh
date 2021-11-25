#!/bin/bash

cd ..

export NUM_OF_WORKERS=32
export AGGR_TREE_BATCH=2
export UPLOAD_BATCH_NUM=100
export AGGREGATOR_SVC_NAME="aggregator"
export FRONTEND_SVC_NAME="frontend"
export STORAGE_SVC_NAME="storage"
export MONGODB_IP="mongodb://10.240.0.4:27017"
export SECUREML_NAMESPACE="secureml"
export GIT_USERNAME=""
export GIT_PASSWORD=""
export DOCKER_CONFIG_JSON=""
export COMMIT_HASH=""

export STORAGE_NODE_HOSTNAME=""
export AGGREGATE_NODE_HOSTNAME=""
export FRONTEND_NODE_HOSTNAMES=""

cd deploy

echo "NUM_OF_WORKERS = ${NUM_OF_WORKERS}"
echo "AGGR_TREE_BATCH = ${AGGR_TREE_BATCH}"
echo "UPLOAD_BATCH_NUM = ${UPLOAD_BATCH_NUM}"
echo "AGGREGATOR_SVC_NAME = ${AGGREGATOR_SVC_NAME}"
echo "FRONTEND_SVC_NAME = ${FRONTEND_SVC_NAME}"
echo "STORAGE_NODE_IP = ${STORAGE_NODE_IP}"
echo "STORAGE_SVC_NAME = ${STORAGE_SVC_NAME}"
echo "MONGODB_IP = ${MONGODB_IP}"
echo "SECUREML_NAMESPACE = ${SECUREML_NAMESPACE}"
echo ""

function run_yaml_template() {
  echo "[run_yaml_template] apply yaml $1"
  cat $1 | envsubst | kubectl apply -f -
  echo ""
}

function stop_yaml_template() {
  echo "[stop_yaml_template] delete yaml $1"
  cat $1 | envsubst | kubectl delete -f -
  echo ""
}

function run_aggregator() {
  if [[ $1 = "stop" ]]; then
    echo "[aggregator] stop aggregator ..."
    stop_yaml_template ./templates/aggregator.yaml
    echo ""
  else
    echo "[aggregator] run aggregator ..."
    run_yaml_template ./templates/aggregator.yaml
    echo ""
  fi
}

function run_frontend() {
  if [[ $1 = "stop" ]]; then
    echo "[frontend] stop frontend ..."
    stop_yaml_template ./templates/frontend.yaml
    echo ""
  else
    echo "[frontend] run frontend ..."
    run_yaml_template ./templates/frontend.yaml
    echo ""
  fi
}

function run_worker() {
  if [[ $1 = "stop" ]]; then
    echo "[worker] stop worker ..."
    stop_yaml_template ./templates/worker.yaml
    echo ""
  else
    echo "[worker] run worker ..."
    run_yaml_template ./templates/worker.yaml
    echo ""
  fi
}

function run_storage() {
  if [[ $1 = "stop" ]]; then
    echo "[storage] stop storage ..."
    stop_yaml_template ./templates/storage.yaml
    echo ""
  else
    echo "[storage] run storage ..."
    run_yaml_template ./templates/storage.yaml
    echo ""
  fi
}

function run_upload_mnist() {
     if [[ $1 = "stop" ]]; then
     echo "[storage] stop upload_mnist ..."
     stop_yaml_template ./templates/upload_mnist.yaml
     echo "[storage] stop upload_mnist_data ..."
     stop_yaml_template ./templates/upload_mnist_data.yaml
     echo ""
   else
     echo "[storage] run upload_mnist ..."
     run_yaml_template ./templates/upload_mnist.yaml
     echo "[storage] run upload_mnist_data ..."
     run_yaml_template ./templates/upload_mnist_data.yaml
     echo ""
   fi
 }

function init() {
  echo "[init] create initial namespace"
  kubectl create namespace ${SECUREML_NAMESPACE}
  echo ""

  echo "[init] create mysecret"
  run_yaml_template ./templates/secrets.yaml 
  echo ""
}

case $1 in
  aggregator)
    init
    run_aggregator $2
    ;;

  frontend)
    init
    run_frontend $2
    ;;

  worker)
    run_worker $2
    ;;

  storage)
    init
    run_storage $2
    ;;

  upload_mnist)
    init
    run_storage $2
    sleep 10
    run_upload_mnist $2
    ;;

  all)
    run_aggregator $2
    sleep 2
    run_frontend $2
    sleep 2
    run_worker $2
    ;;
esac

