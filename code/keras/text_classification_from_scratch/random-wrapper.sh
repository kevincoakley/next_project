#!/bin/bash

set -e

# set TMPDIR variable
export TMPDIR=$_CONDOR_SCRATCH_DIR

echo
echo "I'm running on" $(hostname -f)
echo "OSG site: $OSG_SITE_NAME"
echo

if [[ "$1" != "" ]]; then
    RUN_NAME="$1"
else
    RUN_NAME="not_set"
fi

echo "RUN NAME: $RUN_NAME"

if [[ ! -v OSG_MACHINE_GPUS ]]; then
    echo "OSG_MACHINE_GPUS is not set"
elif [[ -z "$OSG_MACHINE_GPUS" ]]; then
    echo "OSG_MACHINE_GPUS is set to the empty string"
elif [[ $OSG_MACHINE_GPUS == "0" ]]; then
    echo "OSG_MACHINE_GPUS has the value: $OSG_MACHINE_GPUS"
else
    echo "OSG_MACHINE_GPUS has the value: $OSG_MACHINE_GPUS"
    echo "Running nvidia-smi"
    nvidia-smi
fi

if [ -e aclImdb_v1.tar.gz ]
then
    echo "aclImdb_v1.tar.gz found"
    tar -xf aclImdb_v1.tar.gz
    rm -r aclImdb/train/unsup
else
    echo "aclImdb_v1.tar.gz not found"
fi

python ./text_classification_from_scratch.py --num-runs 100 --run-name $RUN_NAME 

tar zcvf text_classification_from_scratch.tar.gz text_classification_from_scratch*
