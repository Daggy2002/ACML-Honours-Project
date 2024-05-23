#!/bin/bash

# This script is used to submit the batch files to the cluster

# run `sbatch job1.batch` 15 times, with a 1 second delay between each submission

for i in {1..15}
do
    sbatch job.batch
    sleep 1
done