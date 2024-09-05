# Linux (or WSL) Python setup environment for Machine learning Laboratory

This is how I set up a fresh linux installation to start working in machine learning and programming. 

It is also useful for WSL (Windows Subsystem for Linux) and I will add comments for it as well.

I keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.

<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [WSL installation guide](#wsl-installation-guide)
    - [WSL: Paths to directories outside of the Linux environment](#wsl-paths-to-directories-outside-of-the-linux-environment)
    - [WSL: About .profile and .bash_profile](#wsl-about-profile-and-bash_profile)
- [Basic Settings](#basic-settings)
    - [Install basic apt and apt-get software](#install-basic-apt-and-apt-get-software)
    - [Install SublimeText](#install-sublimetext)
        - [Easy TeX math: Add paired $ signs to the keybinds](#easy-tex-math-add-paired--signs-to-the-keybinds)
    - [Easily transform 2 spaced indent to 4 spaced indent](#easily-transform-2-spaced-indent-to-4-spaced-indent)
- [Setup Git](#setup-git)
    - [Check your branches in git log history in a pretty line](#check-your-branches-in-git-log-history-in-a-pretty-line)
    - [Push with tags: multi-line git alias](#push-with-tags-multi-line-git-alias)
    - [GitHub Markdown math expressions for README.md, etc.](#github-markdown-math-expressions-for-readmemd-etc)
    - [GitLab Markdown math expressions for README.md, etc.](#gitlab-markdown-math-expressions-for-readmemd-etc)
    - [Install Git Large File System](#install-git-large-file-system)
    - [Make a new Git \(LFS\) repository from local](#make-a-new-git-lfs-repository-from-local)
    - [Manage multiple GitHub or GitLab accounts](#manage-multiple-github-or-gitlab-accounts)
        - [WSL and Windows shared ssh keys for multiple git accounts](#wsl-and-windows-shared-ssh-keys-for-multiple-git-accounts)
- [Install Docker Engine for Linux.](#install-docker-engine-for-linux)
- [Install Python versions with pyenv and virtual environments with poetry](#install-python-versions-with-pyenv-and-virtual-environments-with-poetry)
    - [Useful Data Science libraries](#useful-data-science-libraries)
        - [Basic tasks:](#basic-tasks)
        - [Plotting:](#plotting)
        - [Basic data science and machine learning:](#basic-data-science-and-machine-learning)
        - [Data mining / text mining / crawling / scraping websites:](#data-mining--text-mining--crawling--scraping-websites)
        - [Natural language processing \(NLP\):](#natural-language-processing-nlp)
        - [Neural network and machine learning:](#neural-network-and-machine-learning)
        - [XGBoost](#xgboost)
        - [LightGBM](#lightgbm)
        - [MINEPY / Maximal Information Coefficient](#minepy--maximal-information-coefficient)
        - [Computer Vision \(OpenCV\)](#computer-vision-opencv)
            - [Install OpenCV with ffmpeg](#install-opencv-with-ffmpeg)
- [Install and setup Flask for Python Web Development](#install-and-setup-flask-for-python-web-development)
- [Shell Scripting for convenience](#shell-scripting-for-convenience)
    - [Basic flag setup with getopts](#basic-flag-setup-with-getopts)
    - [Argparse-bash by nhoffman](#argparse-bash-by-nhoffman)
- [CUDA and GPU settings](#cuda-and-gpu-settings)

<!-- /MarkdownTOC -->

<a id="wsl-installation-guide"></a>
## WSL installation guide

https://www.groovypost.com/howto/install-windows-subsystem-for-linux-in-windows-11/

1. Open the cmd with administrator privileges
2. `wsl --install`
3. Restart computer
4. Make username under new Linux terminal

<a id="wsl-paths-to-directories-outside-of-the-linux-environment"></a>
### WSL: Paths to directories outside of the Linux environment

If in the above tutorial for separate git accounts, for example, you needed to use paths to locations in the Windows system, you can replace C: with /mnt/c/

<a id="wsl-about-profile-and-bash_profile"></a>
### WSL: About .profile and .bash_profile

There is a difference between running an interactive shell inside of a started up Linux system, say, if you had Ubuntu installed the regular way, and running the main WSL window. 

Running the WSL software inside Windows opens a login shell, which is different from the interactive shell we are used to.

Login shells load .profile, which then reads .bashrc if the shell being used is bash. However, .profile is ignored if there exists a .bash_profile, which means usually that .bashrc will never be read, and so will also .bash_aliases not be read.

This can be fixed in a few ways:

1. Run the command `bash` every time at start up to open an interactive shell inside the login shell. The only difference this makes is that to exit WSL via commands you'd have to run `exit` on the interactive shell and then on the login shell, twice.

2. Move all the contents of .bash_profile to .profile, then delete .bash_profile so that there's nothing stopping all the initial codes from running even in a login shell.

3. Add `source ~/.profile` to the beginning of .bash_profile so that it is run regardless, and therefore also loads .bashrc if necessary. Personally I chose this one.

<a id="basic-settings"></a>
## Basic Settings

Setup root password:
https://www.cyberciti.biz/faq/how-to-change-root-password-on-macos-unix-using-terminal/

```
sudo passwd root
```

- Set up the screen lock command so you can do it every time you stand up:
https://itsfoss.com/ubuntu-shortcuts/

- Set up your WiFi connection.
- For ease of use, make hidden files and file extensions visible.

Hidden files visibility:
I can't work without seeing hidden files, so in Ubuntu we can do `CTRL+H` and the hidden files will appear. 

To set it as the default:

https://help.ubuntu.com/stable/ubuntu-help/nautilus-views.html.en

<a id="install-basic-apt-and-apt-get-software"></a>
### Install basic apt and apt-get software

In order for most anything else to install properly, we need these first:

```
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


<a id="install-docker-engine-for-linux"></a>
## Install Docker Engine for Linux.

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/engine/install/

For Ubuntu, specifically, there's this guide:
https://docs.docker.com/engine/install/ubuntu/

Reboot after installing.


<a id="cuda-and-gpu-settings"></a>
## CUDA and GPU settings

For a linux server to use an nvidia GPU for calculations instead of the CPU, we need to install CUDA, and for neural networks specifically, cuDNN is also needed.

1. NVIDIA drivers installation guide:  
    https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html


2. CUDA Installation guide:  
    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

3. cuDNN installation guide:  
    https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download

If you already have nvidia or cuda tools installed and want to upgrade, you can use these commands before installing the new software:

```
apt remove --purge cuda*
apt remove --purge nvidia*
apt remove --purge libcuda*
```

Be prepared to run your server in console only mode, since you'll be altering the graphics drivers during this process.

However I personally prefer to use the [`nvidia/cuda` docker images](https://hub.docker.com/r/nvidia/cuda)

---

That is all for now. This is my initial setup for the lab environment under a proxy. If I have any projects that need further tinkering, that goes on another repository / tutorial.

