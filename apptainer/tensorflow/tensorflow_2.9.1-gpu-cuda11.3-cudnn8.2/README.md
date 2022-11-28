# Docker and Singularity Build Instructions

    sudo docker build --build-arg TF_PACKAGE_VERSION=2.9.1 .
    sudo docker images
    sudo docker save <image id> -o tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.tar
    sudo singularity build tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.sif docker-archive://tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.tar