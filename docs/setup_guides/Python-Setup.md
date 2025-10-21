# Python Setup

## pyenv / pyenv-win

[Pyenv](https://github.com/pyenv/pyenv) is a Python version manager that allows you to isolate different installations of python in the same system. It also makes it easy to uninstall without unintended consequences when you need to start from fresh for whatever reason.

Depending on your installation, you might already have a python, but it is better to avoid using it as it interacts with the system, so we install a local version with Pyenv. Pyenv also makes it so that pip and python are always matched for each other in the correct version, which can be a headache if you don't install them isolated.

This is specially useful if you need different versions for different projects (Maybe caused by incompatible updates).

There's also a fork that works on windows called [`pyenv-win`](https://github.com/pyenv-win/pyenv-win), although I entirely recommend just working in WSL instead of natively on Windows to avoid these kinds of issues.

### pyenv for Linux and MacOSX

Sources:
- [Installation guide in pyenv README](https://github.com/pyenv/pyenv#installation)

For Linux and MacOSX natively, and for Linux on WSL on Windows, we have to use the github distribution, clone it with `depth=1` to avoid downloading unnecessary past commits, and install it.

Then we add the paths to `.bash_profile` for bash or to `.zprofile` for zsh.

```sh
# @ shell(linux/mac_osx/wsl)

cd ~
git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
cd ~
source ~/.bash_profile
```

And then we can add it to our PATH so that every time we open `python` it's the pyenv one and not the system one:

```sh
# @ shell(linux/mac_osx/wsl) bash

echo '' >> ~/.bash_profile
echo '# Install pyenv to ~/.pyenv' >> ~/.bash_profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile

source ~/.bash_profile
```
In case you use fish:

```sh
# @ shell(linux/mac_osx/wsl) fish
set -Ux PYENV_ROOT $HOME/.pyenv
test -d $PYENV_ROOT/bin; and fish_add_path $PYENV_ROOT/bin

echo '' >> ~/.config/fish/config.fish
echo '# Install pyenv to ~/.pyenv' >> ~/.config/fish/config.fish
echo 'pyenv init - fish | source' >> ~/.config/fish/config.fish

source ~/.config/fish/config.fish
```


For Dockerfiles:
```Dockerfile
# @ dockerfile

USER root
ENV HOME="/root"

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG C.UTF-8

# avoid tzdata from interrupting build to ask location
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata; \
    apt-get clean

# runtime dependencies
# pyenv see https://github.com/pyenv/pyenv/wiki#troubleshooting--faq
# plus common utilities
RUN apt update; \
    apt install -y --no-install-recommends \
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
        libxml12-dev \
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
        ; \
    rm -rf /var/lib/apt/lists/*

# install pyenv and python 3.11.13
# as of 2024-09, pytorch supports 3.12 but tensorflow seems to have some issues
# https://github.com/tensorflow/tensorflow/issues/62003
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
```


### pyenv-win for Windows

If for some reason you are not using WSL for development (**which I don't recommend**), pyenv-win, which is a fork of pyenv specifically for installing windows versions of python. The benefit of this is it lets you install several versions of python at the same time and use different ones in different projects, without path conflicts.

Let's follow [this installation guide](https://github.com/pyenv-win/pyenv-win#installation)

```sh
# @ git_bash

git clone --depth=1 https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
```

The next step is to add the pyenv paths to the PATH environmental variable so the git bash reads them correctly.

There is two ways of doing this, but I chose option 1 because it's faster and can be copied to new machines.

**Option 1) Adding the paths to .bashrc so they're added each time Git Bash is opened**

```sh
# @ git_bash

nano ~/.bashrc
```

```sh
# @ nano::~/.bashrc

# PYENV paths
export PATH=$PATH:~/.pyenv/pyenv-win/bin:~/.pyenv/pyenv-win/shims
export PYENV=~/.pyenv/pyenv-win/
export PYENV_ROOT=~/.pyenv/pyenv-win/
export PYENV_HOME=~/.pyenv/pyenv-win/
export PYENV_PYTHON_EXE=$(dirname $(pyenv.bat which python))
export PATH=$PATH:$PYENV_PYTHON_EXE
# To update PYENV_PYTHON_EXE if pyenv changes versions close bash and open again

# CTRL+O
# CTRL+X
```

```sh
# @ git_bash

source ~/.bashrc
```


**Option 2) Using Windows Settings**

Go to System Properties, search for Environmental Variables and click the Environmental Variables button that results from an old fashioned settings window.
Under User Environmental Variables add a NEW:

```
Variable name = PYENV
Variable value= %USERPROFILE%\.pyenv\pyenv-win\
```

```
Variable name = PYENV_ROOT
Variable value= %USERPROFILE%\.pyenv\pyenv-win\
```

```
Variable name = PYENV_HOME
Variable value= %USERPROFILE%\.pyenv\pyenv-win\
```

Go to the User Environmental Variables again and select the Path variable, click Edit:

In the edit window, add:
```cmd
%USERPROFILE%\.pyenv\pyenv-win\bin
%USERPROFILE%\.pyenv\pyenv-win\shims
```

Click OK until all the windows go away.


**Restart Git Bash**

```sh
# @ git_bash

echo $PYENV
echo $PATH
```

And the new environmental variables should be added.

Now in Git Bash check if it works

```sh
# @ git_bash

pyenv --version
```

Go to System Properties, search for Environmental Variables and click the Environmental Variables button that results from an old fashioned settings window.

From the User Path variable: delete `%USERPROFILE%\AppData\Local\Microsoft\WindowsApps`, which is conflicting with the `python` command for some reason.

Click OK until all the windows go away.

**Restart Git Bash**

Now if we just run `python` in Git Bash, it will hang instead of opening the interpreter. In order to run Python in Git Bash the same as we did in Unix based systems, [we have to go back to the `.bashrc` and add a path to the a new alias](https://qiita.com/birdwatcher/items/acbc79005d24616de5b6).

```sh
# @ git_bash

nano ~/.bashrc
```

```sh
# @ nano::~/.bashrc

# To run Python in Git Bash like in Unix
alias python='winpty python.exe'

# CTRL+O
# CTRL+X
```

```sh
# @ git_bash

source ~/.bashrc
```

Now typing `python` in Git Bash should open the python interface without hanging.

To update pyenv-win, if installed via Git go to `%USERPROFILE%\.pyenv\pyenv-win` (which is your installed path) and run `git pull`.


### Install Python with pyenv

Now let's install and set the version we will use (`pyenv install -l` to check available versions).

Currently I recommend python 3.12.12, becasue of tensorflow compatibility limitations with higher versions, not to mention there is an available Docker image for 3.12.12 called [3.12-bookworm](https://hub.docker.com/layers/library/python/3.12-bookworm/images/sha256-ef9fc8e04026f71cfb24772366551731703b74282f3519b95baae8618fe47873).

However, with GPU processing projects which use CUDA drivers, I recommend actually using the [Nvidia cuda docker containers](https://hub.docker.com/r/nvidia/cuda) and installing python with pyenv as well.

I also, again, recommend keeping all the python execution on WSL and leaving windows alone, if not for consistency of execution, for the actual reason that most of the time these projects are executed in a Docker container anyways. So while technically you can install python with GitBash, I don't recommend using it for that.


Also, it's worth noting that the installation process uses [some other dependencies we need to install first](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).

Make sure to follow those instructions before continuing, or the installations will fail.

```sh
# @ shell(linux/mac_osx/wsl)

pyenv install 3.12.12
pyenv global 3.12.12
pyenv rehash
```

We can confirm we are using the correct one:

```sh
# @ shell(linux/mac_osx/wsl)

pyenv versions
which python
python --version
which pip
pip --version
```

All these should now return pyenv versions.

Let's also upgrade pip:
```sh
# @ shell(linux/mac_osx/wsl)

pip install --upgrade pip
```

For Dockerfiles (after the code explained for installing pyenv on Docker:
```Dockerfile
# @ dockerfile

USER root
ENV HOME="/root"

RUN pyenv install 3.11.10
RUN pyenv global 3.11.10
RUN pyenv rehash
```

## pipx for isolated installs of CLI python tools

[pipx](https://github.com/pypa/pipx) is a tool that isolates the environment for command line applications that use python. This way, your other projects won't be contaminated with incompatible dependencies without you noticing. It also makes it extremely easy to manage, install and uninstall tools that are not technically to be used as an imported library in this way. You can think of it as the `apt-get install` equivalent for python tools.

Although the official installation guide allows for many ways of installing, I've had trouble running some of these inside Docker images, so I think the best option is actually to install via pip.

Install for Linux/WSL (debian)
```sh
# @ shell(linux/wsl) (debian)

sudo apt-get update
sudo apt-get install pipx
```

Install for Linux/WSL (arch)

```sh
# shell(arch-based)

pacman -Syy
pacman -S python-pipx
```

Install for MacOSX
```sh
# @ shell(mac_osx)

brew install pipx
```

Finish setting up:

```sh
# @ shell(linux/mac_osx/wsl)

pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
```

However, on Dockerfiles, `ensurepath` won't work, so we set the environment variables manually, of course after having installed python via pyenv as explained above.
```Dockerfile
# @ dockerfile
USER root

RUN pip install pipx
ENV PATH="${PATH}:/root/.local/bin"
```

If you are using fish instead of bash do:

```sh
# shell (fish)

fish_add_path '.local/bin'
```

## Poetry for python project and dependency management

Python has been moving away from using `requirements.txt` or `setup.py` to build or install packages, and instead focusing on preparing an independent `pyproject.toml` that specifies build requirements as well as runtime requirements in a way that they can be run automatically. This is outlined in [PEP 517](https://peps.python.org/pep-0517/). Using this building environment, several other packaging tools emerged. While I haven't tested them all, I have chosen Poetry.

[Poetry](https://python-poetry.org/) will not only isolate development environments, but it will recursively check all dependencies, and the dependencies of those dependencies, so that the final installation of a python project doesn't have any incompatible installations. If a package is incompatible for some reason, it will not install that package, and then output the reason. It has saved me from many headaches in the development of many projects. 

It also allows for you to commit the final conclusion of those dependency checks in a file called `poetry.lock`, so that any developer installing the project will have the same environment as the person who did the commit. This, of course, also allows you to go back in a commit history and match the environment to the commit. It also allows for removal of dependencies in a way that will re-check all other packages to see if they can be updated after lifting restrictions. 

I also use the `poetry-plugin-export` plugin to potentially output all of these dependency checks onto a less space heavy requirements.txt so as to use in deploy Docker images, which focus on saving as much space on the image as possible.

### Install poetry with pipx

Poetry isolates every project dependency, but poetry cannot realistically handle its own dependencies in an isolated environment, so we isolate Poetry using `pipx`

For more detail see [Installing poetry with pipx (poetry official website)](https://python-poetry.org/docs/#installing-with-pipx)

I also like installing the export plugin to be able to use poetry for development, and the less resource heavy `requirements.txt` exported by poetry for deploy Docker images.

Install poetry and the export plugin with pipx:

```sh
# @ shell

pipx install poetry && pipx inject poetry poetry-plugin-export
```

In the case of a Dockerfile, of course after having installed python via pyenv as explained above.

```Dockerfile

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry && \
    pipx inject poetry poetry-plugin-export && \
    pipx install toml-cli
```

### Usage

References:
- Follow the [official Poetry documentation](https://python-poetry.org/docs/basic-usage/)
- See [Poetry GitHub issues](https://github.com/python-poetry/poetry/issues) for undocumented problems
- For complicated cases involving GPU or CUDA specific builds, or C++ extensions that are built from source, refer to my personal [Poetry C++ extensions guide](/docs/ai_development/Poetry-C++-extensions-guide.md)


Making a new project can be as easy as:

```sh
# @ shell(linux/mac_osx/wsl)

poetry new project-name-here
cd project-name-here
```

Then, instead of using `pip install` or `pip uninstall` we use `poetry add`. For example, adding the library `numpy` would look like:

```sh
# @ shell(linux/mac_osx/wsl)

poetry add numpy
```

This updates the dependency control files `poetry.toml`, `poetry.lock`, and `pyproject.toml`, which can be committed to version control.

And finally, when cloning a repository, you can use `poetry install` to easily install all the dependencies controlled by poetry in one command.


### Docker + poetry

References:
- [Poetry discussion on Docker best practices](https://github.com/python-poetry/poetry/discussions/1879)


Building poetry projects inside Docker images is incredibly useful, but also very heavy in resources.

That's why I prefer to have two images: a development image, and a deploy image.

The development image can be as heavy as needed, but the deploy image should be lightweight for ease of installing and resource efficiency for the users.

For the deployment image, I recommend a multi stage Dockerfile, where the first images build the poetry environment, and then use the `poetry-plugin-export` plugin to obtain a `requirements.txt` file with all our specifications, which we can then use on the final stage of the Dockerfile.

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
```

For specific examples where I used CUDA compatible images, see my [CUDA python poetry dockerfiles document](../ai_development/CUDA-python-poetry-dockerfiles.md)


## toml-cli for automated edits of the pyproject.toml

Ever since [PEP 518](https://peps.python.org/pep-0518/), accompanied by [PEP 517](https://peps.python.org/pep-0517/), packages are moving from the outdated (and frustratingly varied in behavior) `setup.py` installation to a more standardized installation details format called `pyproject.toml`

Some package management tools like Poetry described above, can handle most of the file automatically and not all of it needs to be edited manually.

However, some details remain to be automated with `poetry ` commands (as of the time of writing 2025-08), mainly because some of it is [not officially supported](https://github.com/python-poetry/poetry/issues/8460). Although this is mostly necessary with libraries that use C extensions for python and is rare, a lot of AI libraries actually need C extensions.

Editing this is trivial in a local environment where we can open a file and edit it directly, but for automatic poetry environment building in Dockerfiles, for example, I try to automate these edits in a way that doesn't imply me having to remember which lines to edit.

For those cases I like using [toml-cli](https://github.com/gnprice/toml-cli), which although not complete, has allowed me to build C dependency projects using poetry inside Docker images.

Like other python-based CLIs, we should install using pipx:

```sh
# @ shell(linux/mac_osx/wsl)

pipx install toml-cli
```
