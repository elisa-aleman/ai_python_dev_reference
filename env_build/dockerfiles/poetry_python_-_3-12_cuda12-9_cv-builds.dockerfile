# CUDA 12.1.0 docker image has been deprecated
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV CUDA_INT=129
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.9
ENV CUDNN_DIR=/opt/cudnn
ENV CUDACXX=/usr/local/cuda/bin/nvcc-12.9

ENV LD_LIBRARY_PATH=/usr/local/lib64
ENV LD_LIBRARY_PATH=$CUDA_TOOLKIT_ROOT_DIR/lib64:$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

ENV HOME="/root"
WORKDIR /root/workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG=C.UTF-8

# avoid tzdata from interrupting build to ask location
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata; \
    apt-get clean

# runtime dependencies
# pyenv see https://github.com/pyenv/pyenv/wiki#troubleshooting--faq
# plus common utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        git \
        libbluetooth-dev \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        llvm \
        make \
        nano \
        tk-dev \
        unzip \
        uuid-dev \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        # usability dependencies for audio and computer vision AI
        ffmpeg \
        g++-12 \
        gcc-12 \
        libgl1 \
        libgomp1 \
        libopencv-dev \
        libprotobuf-dev protobuf-compiler \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# install pyenv and python 3.12.11
# as of 2025-08 pytorch supports 3.13 but Tensorflow only up to 3.12
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install 3.12.11
RUN pyenv global 3.12.11
RUN pyenv rehash

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry && \
    pipx inject poetry poetry-plugin-export && \
    pipx install toml-cli

# Install cmake
# requires libprotobuf-dev protobuf-compiler
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
    tar -zxvf cmake-3.30.4-linux-x86_64.tar.gz && \
    rm cmake-3.30.4-linux-x86_64.tar.gz && \
    mv cmake-* cmake

ENV PATH=/root/cmake/bin:$PATH

CMD ["/bin/bash"]