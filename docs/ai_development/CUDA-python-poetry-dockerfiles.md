# CUDA Python Poetry Dockerfiles

TODO:
- [ ] Add style guide for location comments
- [ ] Add explanation of dockerfiles in env_build/dockerfiles

## Why poetry in docker?

### Development dockerfiles with poetry


### Deploy dockerfiles without poetry


## Setup

### Install docker

Follow the guides to install docker depending on the operating system.

- [Windows / WSL]()
- [Mac OS X]()
- [Linux]()

### Pull the base images

For CUDA 12.1 (best pytorch match):
```sh
# @ shell(linux/mac_osx/wsl)

docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

For CUDA 11.8 (rare dependency incompatibilities):
```sh
# @ shell(linux/mac_osx/wsl)

docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

For more details on pytorch pre-cuda-12 and the specifics of installation with poetry check my [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md) guide.

## Prepare a Dockerfile with the needed dependencies

Under construction.

TODO:
- [ ] Explain that the base images I offer are for libraries that don't need any additional C extensions but can use CUDA
    - [ ] Explain why I wrote each part of this file
- [ ] Link to [Poetry C++ extensions guide](./Poetry-C++-extensions-guide.md) to explain how to build with CUDA for specific libraries even if they don't use poetry

## Build instructions


For CUDA 12.1 (cudnn 8):

```sh
# @ shell(linux/mac_osx/wsl)::/d/git/elisa-aleman/ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-1.dockerfile --tag poetry_python:3.11_cuda12.1_cudnn8 .
```

For CUDA 12.1 (cudnn 8) with computer vision and audio dependencies
```sh
# @ shell(linux/mac_osx/wsl)::/d/git/elisa-aleman/ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-1_cv-builds.dockerfile --tag poetry_python:3.11_cuda12.1_cudnn8_cv-builds .
```


For CUDA 11.8 (cudnn 8):

```sh
# @ shell(linux/mac_osx/wsl)::/d/git/elisa-aleman/ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda11-8.dockerfile --tag poetry_python:3.11_cuda11.8_cudnn8 .
```

For CUDA 11.8 (cudnn 8) with computer vision and audio dependencies
```sh
# @ shell(linux/mac_osx/wsl)::/d/git/elisa-aleman/ai_python_dev_reference

docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda11-8_cv-builds.dockerfile --tag poetry_python:3.11_cuda11.8_cudnn8_cv-builds .
```

### Run Containers with GPU access


TODO:
- [ ] Add build commands with local volume and CUDA device access
- [ ] Explain about container exec without kill exit and lost detaching vs re-attachable with kill exit containers


<!-- 
```sh
# @ shell(linux/mac_osx/wsl)

cd ai_python_dev_reference
docker run -itd --ipc=host -v ${PWD%/*}:/v --shm-size=4gb --gpus ‘“device=1”’ --name ai_dev_example poetry_python:3.11_cuda12.1 bash

docker exec -it -w /v/project ai_dev_example bash

poetry new project-name
cd project-name
nano pyproject.toml
poetry install

exit

touch test.txt
docker exec -it -w /work/project ai_dev_example bash
ls -l

Check UID and GID

chown -R UID:GID ./*
``` -->


## Run with poetry



