# CUDA Python Poetry Dockerfiles

TODO:
- [ ] Add explanation of dockerfiles in env_build/dockerfiles
- [ ] Explain more about container exec without kill exit and lost detaching vs re-attachable with kill exit containers
- [ ] Explain that the base images I offer are for libraries that don't need any additional C extensions but can use CUDA
    - [ ] Explain why I wrote each part of this file
- [ ] Link to [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md) to explain how to build with CUDA for specific libraries even if they don't use poetry
- [ ] Add DeepSeek guide

## Why poetry in docker?

### Development dockerfiles with poetry


### Deploy dockerfiles without poetry


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

For more details on pytorch pre-cuda-12 and the specifics of installation with poetry check my [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md) guide.

## Prepare a Dockerfile with the needed dependencies

*Under construction*



## Build instructions


For CUDA 12.9 (cudnn 9.10):

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-9.dockerfile --tag poetry_python:3.11_cuda12.9_cudnn .
```

For CUDA 12.9 (cudnn 9.10) with computer vision and audio dependencies
```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-9_cv-builds.dockerfile --tag poetry_python:3.11_cuda12.9_cudnn_cv-builds .
```


For CUDA 11.8 (cudnn 8):

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda11-8.dockerfile --tag poetry_python:3.11_cuda11.8_cudnn8 .
```

For CUDA 11.8 (cudnn 8) with computer vision and audio dependencies
```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda11-8_cv-builds.dockerfile --tag poetry_python:3.11_cuda11.8_cudnn8_cv-builds .
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
- `poetry_python:3.11_cuda12.9_cudnn8_cv-builds`: Name of the image we will use. Use whichever you need from the ones built above.
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
    poetry_python:3.11_cuda12.9_cudnn8_cv-builds \
    bash
```

Now, if you are running docker from Docker Desktop for Windows or Mac, even if the user in the Dockerfile is root, the files can still be read and written to by the local host machine. However, if you are running docker on a Linux machine directly (or connecting to a server that runs linux), then the docker image will create files that are exclusively accessed by the local host root user and not local users, unless they `sudo chown -R $(id -u):$(id -g) ./*` on the directory that they need to access that the container wrote to.

The `docker run` can also have the arguments `--user $(id -u):$(id -g)` to give the container the same user and group id as the one we have so that file permissions aren't different, but then it won't have a pre-existing user to match that. 

You can read more about this [on this VSCode doc page about developing on Docker](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user), but have to be careful about the placement considering that the pyenv / python installations on the docker files I provided are only for the root user.

#### Creating a non-root user

Source: 
- [VSCode docs: Creating a non-root user](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)

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

Start a non-re-attachable bash instance (which won't terminate the container if you `exit`). If detached with `CTRL+P+Q`, it will keep running, but we won't have access to the tty again.

```sh
# @ shell(linux/mac_osx/wsl)::/.../ai_python_dev_reference

docker exec -it -w /v/ai_python_dev_reference ai_dev_example bash
```

Or attach to the instance already made (which is re-attachable, but will terminate on `exit`).

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
poetry shell

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
    --python "~3.11" \
    --author "Elisa Aleman <elisa.claire.aleman.carreon@gmail.com>" \
    --license "GPL-3.0-or-later" \
    --no-interaction
mkdir ai_examples
touch ai_examples/__init__.py
touch README.md

poetry add torch==2.4.0 torchvision@* tensorflow-cpu@* onnx@* onnxruntime@* ai_edge_torch
poetry install
poetry shell

# @ ai_dev_example::poetry_shell::/v/ai_python_dev_reference/ai_examples_cuda_12
python
```

More details on the command and info I entered at [the poetry pyproject.toml documentation](https://python-poetry.org/docs/pyproject/)

`poetry install` will install the dependencies stated in the `pyproject.toml` and `poetry.lock` files created when adding projects.