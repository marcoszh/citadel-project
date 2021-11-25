#!/bin/bash

worker_count=0

nodes_list=$(kubectl get pods -n secureml -o=name |  sed "s/^.\{4\}//")

for name in ${nodes_list[@]}; do
    dir="test"
    if [[ $name =~ "aggregator" ]]; then
        # mkdir $dir
        kubectl logs -n secureml $name > aggr.log
    fi
    if [[ $name =~ "worker" ]]; then
        # mkdir $dir
        kubectl logs -n secureml $name > worker$worker_count.log
        (( worker_count++ ))
    fi
done
