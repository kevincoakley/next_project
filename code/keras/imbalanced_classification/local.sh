#!/bin/bash

set -e

if [[ "$1" != "" ]]; then
    CONTAINER_PATH="$1"
else
    exit
fi

apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.06-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.06-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.09-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.09-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.10-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.10-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.11-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.11-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.12-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.12-tf2-py3

apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.8.0-gpu-1.0.0.sif ./fixed-seed-wrapper.sh tensorflow_2.8.0-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.9.1-gpu-1.0.0.sif ./fixed-seed-wrapper.sh tensorflow_2.9.1-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.10.1-gpu-1.0.0.sif ./fixed-seed-wrapper.sh tensorflow_2.10.1-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.11.0-gpu-1.0.0.sif ./fixed-seed-wrapper.sh tensorflow_2.11.0-gpu

apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.06-tf2-py3.sif ./random-wrapper.sh tensorflow_22.06-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.09-tf2-py3.sif ./random-wrapper.sh tensorflow_22.09-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.10-tf2-py3.sif ./random-wrapper.sh tensorflow_22.10-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.11-tf2-py3.sif ./random-wrapper.sh tensorflow_22.11-tf2-py3
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.12-tf2-py3.sif ./random-wrapper.sh tensorflow_22.12-tf2-py3

apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.8.0-gpu-1.0.0.sif ./random-wrapper.sh tensorflow_2.8.0-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.9.1-gpu-1.0.0.sif ./random-wrapper.sh tensorflow_2.9.1-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.10.1-gpu-1.0.0.sif ./random-wrapper.sh tensorflow_2.10.1-gpu
apptainer run --nv $CONTAINER_PATH/docker_hub/tensorflow_2.11.0-gpu-1.0.0.sif ./random-wrapper.sh tensorflow_2.11.0-gpu
