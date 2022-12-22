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

python ./image_classification_from_scratch.py --num-runs 100 --run-name $RUN_NAME 

tar zcvf image_classification_from_scratch.tar.gz image_classification_from_scratch*
