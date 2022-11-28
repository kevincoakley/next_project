# Apptainer Containers 

This directory contains the Apptainer definition files to build the containers.

### Below is a list of the containers and links to download the build containers.

### dockerhub/tensorflow/tensorflow (GPU)

* [tensorflow_2.8.0-gpu](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/tensorflow_2.8.0-gpu-1.0.0.sif)
* [tensorflow_2.9.1-gpu](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/tensorflow_2.9.1-gpu-1.0.0.sif)
* [tensorflow_2.11.0-gpu](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/tensorflow_2.11.0-gpu-1.0.0.sif)

### dockerhub/tensorflow/tensorflow Customize (GPU)

Containers from dockerhub/tensorflow/tensorflow customized to include CUDA 11.3 and cuDNN 8.2.1 due to a bug with the included version of cuDNN and Bi-directional RNN network to return non-deterministic outputs. Build instructions are in the README.md file in the directory.

* [tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/custom/tensorflow_2.8.3-gpu-cuda11.3-cudnn8.2.sif)
* [tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/custom/tensorflow_2.9.1-gpu-cuda11.3-cudnn8.2.sif)
* [tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/docker_hub/custom/tensorflow_2.10.0-gpu-cuda11.3-cudnn8.2.sif)

### nvcr.io/nvidia/tensorflow

* [tensorflow_22.06-tf2-py3](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/ngc/tensorflow_22.06-tf2-py3.sif)
* [tensorflow_22.09-tf2-py3](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/ngc/tensorflow_22.09-tf2-py3.sif)
* [tensorflow_22.10-tf2-py3](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/ngc/tensorflow_22.10-tf2-py3.sif)
* [tensorflow_22.11-tf2-py3](https://object.cloud.sdsc.edu:443/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/singularity/ngc/tensorflow_22.11-tf2-py3.sif)