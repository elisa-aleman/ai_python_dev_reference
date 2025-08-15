# CUDA Python Poetry Dockerfiles

TODO:

- [ ] Add mmdeploy / TensorRT /  ONNXRuntime etc explicit build guide
- [ ] Add DeepSeek guide
- [ ] Add deployment dockerfile tutorial

## Why is CUDA necessary

There's no point in explaining all this before making sure we all know why we are here. CPUs can run our python AI programs just fine, but GPUs run them faster. For nvidia GPUs, there is [CUDA](https://en.wikipedia.org/wiki/CUDA), which is an API that allows programs to access the GPU to run processes (even if they're not particularly graphical). I'm honestly more of a software rather than hardware kind of engineer, and all the different GPUs and their particularities give me a headache, so if you're curious about the specifics I'm afraid you will need to search elsewhere. However, for our purposes, which is running pytorch, tensorflow, onnx, etc. AI models with GPU processing, this guide should suffice.

## Why docker?

You might think to yourself, ah, but I already have a GPU ready with all the CUDA and cuDNN installs necessary all natively in my OS (and cudos to you if you do), but sadly some projects need different versions of CUDA to even run, and you can't really install all of them in the same OS... and if I recall correctly in my college years, updating the CUDA version even if no project of yours will complain, is an entire couple of afternoons levels of complicated. 

Meanwhile, nvidia already has some [nice docker images ready for us to use](https://hub.docker.com/r/nvidia/cuda/tags). So we can just use those.

### Why not the pytorch docker image?

[Have you seen that thing?!](https://hub.docker.com/r/pytorch/pytorch/tags) I can't even begin to state why it is unusable, but I will sure try!

1. They are not releasing the CUDA docker images as fast as the relevant versions are out (as of 2025-08 it seems fine, but a year ago and it was about 5 versions late)
2. They use [conda](https://anaconda.org/anaconda/conda), which, although awesome for what it is, [has some issues](https://medium.com/@digitalpower/comparing-the-best-python-project-managers-46061072bc3f). Namely, it is incompatible with Poetry and PyPI. It also can't be used commercially by large companies, and while I prefer sticking it to corporations, sadly the lack of PyPI compatibility also means they likely don't want to use it anyways. If you, the reader, work for a corporation with more than 200 employees, make sure nobody uses these pytorch docker images.
3. It also locks the python version without stating it clearly.

I personally don't even know why they provide those like that.

## Why poetry in docker?

References:
- [Poetry discussion on Docker best practices](https://github.com/python-poetry/poetry/discussions/1879)


When developing in python in general [Poetry](https://python-poetry.org/), as I explained [in my Python setup guide](../setup_guides/Python-Setup.md#Poetry-for-python-project-and-dependency-management), will recursively check and make sure that all the dependencies installed in the project are not conflicting with each other, which Python's default dependency installer `pip` does nothing to stop. This can lead to problems during the execution of any programs you might be writing. When preparing a docker image, one really wants to make sure that things run smoothly anywhere that the docker image will be run, since that's the main purpose of docker in general. So it stands to reason that we want to make sure that the docker image we want to distribute has no dependency incompatibilities.

However, poetry itself and the development environment it provides can make a Docker image heavier than it needs to be. That's why I prefer to have two images: a development image, and a deploy image.


### Development dockerfiles with poetry

Development dockerfiles have no weight restrictions, but there's a few details in installing and running it that are not obvious at first so I'll explain them here.

I already prepared some completed dockerfiles in [`ai_python_dev_reference/env_build/dockerfiles/`](../../env_build/dockerfiles/) that we can use. This section will explain them more in depth. 

*Note: although, as the time of writing 2025-08 CUDA 13 is not compatible with most of the tools we use*

First we need to use an image with CUDA and cudnn provided by nvidia.

At the time of writing (2025-08) the latest compatible CUDA image and python version are CUDA 12.9 and python 3.12.11

You can check for yourself in the [Python readiness matrix](https://pyreadiness.org) which covers `torch` up to 3.13, but also in the [Pytorch website](https://pytorch.org/), in the [Tensorflow website](https://www.tensorflow.org/install), which covers up to 3.12, [the ONNXRuntime releases](https://github.com/microsoft/onnxruntime/releases), which covers up to python 3.12

Having determined that, we setup the base image as well as some environment variables and our working directory. These will be used by other libraries we might install, and we don't want to be typing it all the time.

```dockerfile
# @ ai_python_dev_reference/env_build/dockerfiles/poetry_python_-_3-12_cuda12-9.dockerfile

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
```

Then we need to provide the basic dependencies for anything to be installed, really, and these are often a headache since they are not in any damn tutorial you find for most stuff, and for convenience of maintaining docker images light, the one provided by NVIDIA does not have any of these installed.

However, because Docker will need to run the image building automatically, we have to disable any interactivity from the commonly used shell commands.

```dockerfile
# @ ai_python_dev_reference/env_build/poetry_python_-_3-12_cuda12-9.dockerfile

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
    && rm -rf /var/lib/apt/lists/*

```

Finally we install python using pyenv to avoid locking ourselves to one single python version once we run the container. Even though we most likely want to, for consistency, it's a real pain in the ass to find out you need another python version to test something and see that you can't override anything because it's already locked in. It's just good practice, anyways. The deployment image can have a locked python version if you so prefer.

Since we can't really reboot the shell we do `pyenv rehash` at the end.

```dockerfile
# @ ai_python_dev_reference/env_build/poetry_python_-_3-12_cuda12-9.dockerfile

# install pyenv and python 3.12.11
# as of 2025-08 pytorch supports 3.13 but Tensorflow only up to 3.12
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install 3.12.11
RUN pyenv global 3.12.11
RUN pyenv rehash
```

And finally, we can write the lines that will install `pipx`, `poetry`, and `toml-cli`, and finish it up with the `CMD` to run [when the container is run](https://docs.docker.com/reference/dockerfile/).

```dockerfile
# @ ai_python_dev_reference/env_build/poetry_python_-_3-12_cuda12-9.dockerfile

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry && \
    pipx inject poetry poetry-plugin-export && \
    pipx install toml-cli

CMD ["/bin/bash"]
```

In the case we need to use audio input or output for the AI models, or for computer vision models that process images or video, we need to add the following dependencies, as well as get cmake for most C extensions we might need to build.

```dockerfile
# @ ai_python_dev_reference/env_build/poetry_python_-_3-12_cuda12-9_cv-builds.dockerfile

RUN apt-get update && apt-get install -y --no-install-recommends \
        # ... other dependencies here
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

# Install cmake
# requires libprotobuf-dev protobuf-compiler
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
    tar -zxvf cmake-3.30.4-linux-x86_64.tar.gz && \
    rm cmake-3.30.4-linux-x86_64.tar.gz && \
    mv cmake-* cmake
```

If you only need python libraries that don't depend on C extensions but can use CUDA already as packaged in PIPy, then you can use the dockerfiles I provide as they are.

Now you might need to extend these dockerfiles. For example if you want to add [TensorRT](https://developer.nvidia.com/tensorrt), [TorchScript](https://docs.pytorch.org/docs/stable/cpp_index.html) or [ONNXRuntime](https://onnxruntime.ai/docs/extensions/) C extensions (although this might not be the case anymore with PyPI `onnxruntime-extensions`), these need to be downloaded built and installed separate to poetry and then added as a local dependency. For more details and examples, check my [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md).


### Deploy dockerfiles without poetry

*Under construction*



For the deployment image, I recommend a multi stage Dockerfile, where the first images build the poetry environment, and then use the `poetry-plugin-export` plugin to obtain a `requirements.txt` file with all our specifications, which we can then use on the final stage of the Dockerfile.

<!-- 
```Dockerfile
# Generate workable requirements.txt from Poetry dependencies 
FROM python:3-slim as requirements 

RUN apt-get install -y --no-install-recommends build-essential gcc
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry==1.7.1 && pipx inject poetry poetry-plugin-export

COPY pyproject.toml poetry.lock ./ 
RUN poetry export -f requirements.txt --without-hashes -o /src/requirements.txt 

# Final app image 
FROM python:3-slim as webapp 

# Switching to non-root user appuser 
RUN adduser appuser 
WORKDIR /home/appuser 
USER appuser:appuser 

# Install requirements 
COPY --from=requirements /src/requirements.txt . 
RUN pip install --no-cache-dir --user -r requirements.txt
``` -->


## Setup

### Install docker

Follow the guides to install docker depending on the operating system.

- [Windows / WSL](../setup_guides/Windows-Setup.md#Install-Docker-Desktop-for-Windows)
- [Mac OS X](../setup_guides/MacOS-Setup.md#Install-Docker-Desktop-for-Mac)
- [Linux](../setup_guides/Linux-WSL-Setup.md#Install-Docker-Engine-for-Linux)


### Pull the base images

For CUDA 12.9 (best pytorch match):
```sh
# @ shell(linux/mac_osx/wsl)

docker pull nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
```

For CUDA 11.8 (rare dependency incompatibilities):
```sh
# @ shell(linux/mac_osx/wsl)

docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

For more details on pytorch pre-cuda-12 and the specifics of installation with poetry check my [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md).


## Prepare Dockerfile with the needed dependencies

You can use one of the Dockerfiles provided in [`ai_python_dev_reference/env_build/dockerfiles/`](../../env_build/dockerfiles/) as a base, or write your own.


- `poetry_python_-_3-12_cuda12-9.dockerfile`
    - For CUDA 12.9 (cudnn 9.10)
- **`poetry_python_-_3-12_cuda12-9_cv-builds.dockerfile`**
    - For CUDA 12.9 (cudnn 9.10) with computer vision and audio dependencies
    - **I recommend this one** if you don't mind a bit extra weight.
- `poetry_python_-_3-12_cuda11-8.dockerfile`
    - For CUDA 11.8 (cudnn 8)
    - Some projects with dependencies like [CTranslate2](https://github.com/OpenNMT/CTranslate2) were late to update to CUDA 12 (Feb 2024), but now (2025-08) it seems it's not an issue, so stick with CUDA 12.
- `poetry_python_-_3-12_cuda11-8_cv-builds.dockerfile`
    - For CUDA 11.8 (cudnn 8) with computer vision and audio dependencies
    - Same advice as above, stick with CUDA 12 unless really necessary.
- `poetry_python_-_3-12_cuda13-0.dockerfile`
    - For CUDA 13.0 (cudnn 9.12)
    - Still not compatible with most anything relevant for AI as of 2025-08.
- `poetry_python_-_3-12_cuda13-0_cv-builds.dockerfile`
    - For CUDA 13.0 (cudnn 9.12) with computer vision and audio dependencies
    - Still not compatible with most anything relevant for AI as of 2025-08.


Now you might need to extend these files for C extensions, as explained above. Again, for more details see my [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md).

## Build instructions


For CUDA 12.9 (cudnn 9.10):

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-12_cuda12-9.dockerfile --tag poetry_python:3.12_cuda12.9_cudnn .
```

For CUDA 12.9 (cudnn 9.10) with computer vision and audio dependencies
```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-12_cuda12-9_cv-builds.dockerfile --tag poetry_python:3.12_cuda12.9_cudnn_cv-builds .
```


### Run Containers with GPU access

Start a container with GPU access and mounted volume access to your project.

We want to achieve a space we can treat as a closed environment but still allow us to navigate files and execute code as if it were our own terminal.

For `docker run`:

- `--interactive`: keeps the STDIN open for us to enter inputs
- `--tty`: Adds a pseudo-terminal to the container to be able to run those inputs
- `--detach`: starts the container in a detached state, so we can attach to it in different ways as we want later on.
- `--ipc=host`: Opens memory access to be shared with the host. [Specially useful for multiprocessing and multithreaded data loaders used in pytorch](https://github.com/pytorch/pytorch#docker-image). However, it should not be used if there are security concerns.
- `-shm-size=4gb`: The amount of shared memory. Restrict as needed.
- `--gpus all`: Gives access to all GPUs available. Restrict as needed.
- `--volume ${PWD%/*}:/v`: The parent directory from our current location is expressed by `${PWD%/*}`. This gives access to the container to our current project and others in the same parent directory, in case we need to create other poetry projects with the same container.
- `--name ai_dev_example`: Name of the container. Use what you like.
- `poetry_python:3.12_cuda12.9_cudnn8_cv-builds`: Name of the image we will use. Use whichever you need from the ones built above.
- `bash`: Starts a login shell on top of the shell made by the image.

For more details, check the [docker run documentation](https://docs.docker.com/reference/cli/docker/container/run/)


```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker run \
    --interactive \
    --tty \
    --detach \
    --ipc=host \
    --shm-size=4gb \
    --gpus all \
    --volume ${PWD%/*}:/v \
    --name ai_dev_example \
    poetry_python:3.12_cuda12.9_cudnn_cv-builds \
    bash
```

Now, if you are running docker from Docker Desktop for Windows or Mac, even if the user in the Dockerfile is root, the files can still be read and written to by the local host machine. However, if you are running docker on a Linux machine directly (or connecting to a server that runs linux), then the docker image will create files that are exclusively accessed by the local host root user and not local users, unless they `sudo chown -R $(id -u):$(id -g) ./*` on the directory that they need to access that the container wrote to.

The `docker run` can also have the arguments `--user $(id -u):$(id -g)` to give the container the same user and group id as the one we have so that file permissions aren't different, but then it won't have a pre-existing user to match that. 

You can read more about this [on this VSCode doc page about developing on Docker](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user), but have to be careful about the placement considering that the pyenv / python installations on the docker files I provided are only for the root user.

#### Creating a non-root user

Source: 
- [VSCode docs: Creating a non-root user](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)

I leave this advice here, although my provided dockerfiles don't follow it for simplicity. I leave it to the reader to determine if this is relevant to their situation, particularly if in a company production environment.


> While any images or Dockerfiles that come from the Dev Containers extension will include a non-root user with a UID/GID of 1000 (typically either called vscode or node), many base images and Dockerfiles do not. Fortunately, you can update or create a Dockerfile that adds a non-root user into your container.
> 
> Running your application as a non-root user is recommended even in production (since it is more secure), so this is a good idea even if you're reusing an existing Dockerfile. For example, this snippet for a Debian/Ubuntu container will create a user called user-name-goes-here, give it the ability to use sudo, and set it as the default:

```Dockerfile
ARG USERNAME=user-name-goes-here
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
```



### Attach to container

*Under construction*

There are two main options to attach to the now running container.

1. Non-re-attachable bash new instance that will keep the container running on exit (think of it as exiting the bash, not the container)
2. Re-attachable bash that's already running from the run command that will shut down the container processes on exit (think of it as the main CMD started by the Dockerfile)

For option 1: Start a non-re-attachable bash instance, which won't terminate the container if you `exit`. If detached with `CTRL+P+Q`, it will keep running, but we won't have access to the tty again. Instead, if we run another bash instance it will attach to a new tty, without any command history or output from before exiting.

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker exec -it -w /v/ai_python_dev_reference ai_dev_example bash
```

Or option 2: attach to the instance already made (which is re-attachable, but will terminate on `exit`).

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker attach ai_dev_example
```

Do consider this last one has a different working directory.


## Run with poetry

In my case I already have a poetry example made:

```sh
# @ ai_dev_example::/v/ai_python_dev_reference
cd ai_examples_cuda_12
eval $(poetry env activate)

# @ ai_dev_example::poetry_shell::/v/ai_python_dev_reference/ai_examples_cuda_12
poetry install
python
```

But in any other case you can make your own, this is what I did for the above example:

```sh
# @ ai_dev_example::/v/ai_python_dev_reference
mkdir ai_examples_cuda_12
cd ai_examples_cuda_12
poetry init \
    --name "ai_examples" \
    --description "AI model example code for training and quantization, as well as converting, pytorch, onnx, tflite, and ai_edge_torch" \
    --python "~3.12" \
    --author "Elisa Aleman <elisa.claire.aleman.carreon@gmail.com>" \
    --license "GPL-3.0-or-later" \
    --no-interaction
mkdir ai_examples
touch ai_examples/__init__.py
touch README.md

poetry add ai_edge_torch@0.4.0 torch@2.6.0 tensorflow@2.19.1
poetry add torchvision@0.21.0
poetry add onnx onnxruntime-gpu

poetry install
eval $(poetry env activate)

# @ ai_dev_example::poetry_shell::/v/ai_python_dev_reference/ai_examples_cuda_12
python
```

More details on the command and info I entered at [the poetry pyproject.toml documentation](https://python-poetry.org/docs/pyproject/)

`poetry install` will install the dependencies stated in the `pyproject.toml` and `poetry.lock` files created when adding projects.