FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# fix Nvidia keys issue https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install Python and other needed packages
RUN apt-get update && apt-get install -y ffmpeg unzip wget libopenmpi-dev libosmesa6-dev patchelf libsm6 libxext6 python3.7  python3.7-dev python3-pip && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip

# Install Requirements
ADD /docker/requirements.txt /home/requirements.txt
RUN python3 -m pip install -r /home/requirements.txt
