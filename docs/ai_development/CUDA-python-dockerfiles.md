
```
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# reference python:3.11.7-bookworm
ENV LANG C.UTF-8

# avoid tzdata from interrupting build to ask location
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata; \
    apt-get clean

# runtime dependencies
RUN apt update; \
    apt install -y --no-install-recommends \
        build-essential curl git libbluetooth-dev libbz2-dev libffi-dev liblzma-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml12-dev libxmlsec1-dev llvm make tk-dev uuid-dev wget xz-utils zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*

ENV HOME="/root"
WORKDIR $HOME

# install pyenv and python 3.11.7
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
EMV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install 3.11.7
RUN pyenv global 3.11.7
RUN pyenv rehash

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry==1.7.1 && pipx inject poetry poetry-plugin-export
pipx install toml-cli

CMD ["/bin/bash"]
```



```
docker build --no-cache -t poetry_python:3.11.7-bookworm - << EOF
FROM python:3.1.1.7-bookworm
RUN apt update && \
    apt install -y pipx
ENV PATH=“$PATH:/root/.local/bin”
RUN pipx install poetry==1.7.1 && pipx inject poetry poetry-plugin-export
CMD[“/bin/bash”]
EOF

docker run -itd --ipc=host -v ${PWD%/*}:/work --shm-size=4gb --gpus ‘“device=1”’ --name my_container python:3.11.7-bookworm bash

docker exec -it -w /work/project my_container bash

poetry new project-name
cd project-name
nano pyproject.toml
poetry install

exit

touch test.txt
docker exec -it -w /work/project my_container bash
ls -l

Check UID and GID

chown -R UID:GID ./*
```
