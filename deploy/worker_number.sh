#!/bin/bash

cd /home/SecureML/AdminEnclave

count=3
NUM_OF_WORKERS=$1
NUM_THRESHOLD=$((NUM_OF_WORKERS-2))
while [ $count -le $NUM_THRESHOLD ];
do
    write_count=$((count+1))
    sed -ie "/$count: b.*/a\ \ \ \ $write_count: b\"needstobesixteen\"," frontend.py
    ((count=count+1))
done