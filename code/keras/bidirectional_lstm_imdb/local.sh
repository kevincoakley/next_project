#!/bin/bash

set -e

if [[ "$1" != "" ]]; then
    CONTAINER_PATH="$1"
else
    exit
fi

apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.06-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.06-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.09-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.09-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.10-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.10-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.11-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.11-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.12-tf2-py3.sif ./fixed-seed-wrapper.sh tensorflow_22.12-tf2

apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2.sif ./fixed-seed-wrapper.sh tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.sif ./fixed-seed-wrapper.sh tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2.sif ./fixed-seed-wrapper.sh tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6.sif ./fixed-seed-wrapper.sh tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6

apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.06-tf2-py3.sif ./random-wrapper.sh tensorflow_22.06-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.09-tf2-py3.sif ./random-wrapper.sh tensorflow_22.09-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.10-tf2-py3.sif ./random-wrapper.sh tensorflow_22.10-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.11-tf2-py3.sif ./random-wrapper.sh tensorflow_22.11-tf2
apptainer run --nv $CONTAINER_PATH/ngc/tensorflow_22.12-tf2-py3.sif ./random-wrapper.sh tensorflow_22.12-tf2

apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2.sif ./random-wrapper.sh tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.sif ./random-wrapper.sh tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2.sif ./random-wrapper.sh tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2
apptainer run --nv $CONTAINER_PATH/docker_hub/custom/tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6.sif ./random-wrapper.sh tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6
