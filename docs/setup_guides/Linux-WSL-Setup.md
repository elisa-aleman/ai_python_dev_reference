# Linux (or WSL) setup environment for development

This guide is under construction

TODO:
- [ ] Re-check old narrative
- [ ] Add comments about proxy?
- [ ] Check all syntax highlights and location comments
- [ ] Wrap all links
- [ ] propagate relevant parts of suggested tools guide to OS specific guides with links to this main document

This is how I set up a fresh linux installation to start working in machine learning and programming. 

It is also useful for WSL (Windows Subsystem for Linux) and I will add comments for it as well.

I keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.


## WSL installation guide

If you're using WSL, first read the [guide I wrote for Windows](./Windows-Setup.md#wsl-windows-subsystem-for-linux-installation-guide)

Remember that to run docker in WSL, it has to be installed in Windows using Docker Desktop.

## Terminal

Refer to my section in [Suggested Tools and Setup: GNU Terminal for Linux](./Suggested-Tools-and-Setup.md#gnu-terminal-for-linux)


## Basic Settings

References:
- [How to change root password on macos unix using terminal](https://www.cyberciti.biz/faq/how-to-change-root-password-on-macos-unix-using-terminal/)
- [Ubuntu Shortcuts](https://itsfoss.com/ubuntu-shortcuts/)

Setup root password:

```sh
# @ shell(mac_osx)

sudo passwd root
```

- Set up the screen lock command so you can do it every time you stand up (`Super+L` or `Ctrl+Alt+L`)
- Set up your WiFi connection.
- For ease of use, make hidden files and file extensions visible.

Hidden files visibility:
I can't work without seeing hidden files, so in Ubuntu we can do `CTRL+H` and the hidden files will appear. 

To set it as the default [follow this guide](https://help.ubuntu.com/stable/ubuntu-help/nautilus-views.html.en)



### Install basic apt and apt-get software

In order for most anything else to install properly, we need these first:

```sh
# @ shell(linux)

sudo apt-get update
sudo apt-get install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev
```

Without these, there can sometimes be errors. I include this list here because a lot of the time I would install this and forget, and then struggle with errors when starting from a new machine, or a Docker image.

### Setup git

Follow the [Git Setup and Customization](./Git-Setup-and-Customization.md) for more details.

## Install Docker Engine for Linux

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow [the official guide for docker](https://docs.docker.com/engine/install/)

For Ubuntu, specifically, there's [this guide](https://docs.docker.com/engine/install/ubuntu/).

Reboot after installing.

## Install Python

Follow my [Python setup guide](./Python-Setup.md)

## CUDA and GPU settings

For a linux system to use an nvidia GPU for calculations instead of the CPU, we need to install CUDA, and for neural networks specifically, cuDNN is also needed.

1. [NVIDIA drivers installation guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
2. [CUDA Installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
3. [cuDNN installation guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download)

If you already have nvidia or cuda tools installed and want to upgrade, you can use these commands before installing the new software:

```
apt remove --purge cuda*
apt remove --purge nvidia*
apt remove --purge libcuda*
```

Be prepared to run your server in console only mode, since you'll be altering the graphics drivers during this process.

However I personally prefer to use the [`nvidia/cuda` docker images](https://hub.docker.com/r/nvidia/cuda)

For specific examples where I used CUDA compatible images, see my [CUDA python dockerfiles document](../ai_development/CUDA-python-dockerfiles.md)

---

That is all for now. This is my initial setup for the lab environment under a proxy. If I have any projects that need further tinkering, that goes on another repository / tutorial.

