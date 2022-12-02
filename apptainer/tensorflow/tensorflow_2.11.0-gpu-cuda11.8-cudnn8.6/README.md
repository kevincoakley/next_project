# Docker and Apptainer Build Instructions

    sudo docker build --build-arg TF_PACKAGE_VERSION=2.11.0 .
    sudo docker images
    sudo docker save <image id> -o tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6.tar
    sudo singularity build tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6.sif docker-archive://tensorflow_2.11.0-gpu-cuda11.8-cudnn8.6.tar