# Initial Data Science setup for Windows

I initially used to work on python and Data Science using MacOSX or Ubuntu, so changing to Windows was a confusing experience. Because of this I wrote a guide of everything I did for the initial setup of my computer to start programming.

This is how I set up a fresh windows PC to start working in machine learning and programming. I also keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.

My current Windows PC is using Windows 365.

<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [WSL installation guide](#wsl-installation-guide)
    - [WSL: Paths to directories outside of the Linux environment](#wsl-paths-to-directories-outside-of-the-linux-environment)
    - [WSL: About .profile and .bash_profile](#wsl-about-profile-and-bash_profile)
    - [Linux development guide](#linux-development-guide)
    - [WSL: About DNS issues under VPN or proxy](#wsl-about-dns-issues-under-vpn-or-proxy)
- [Basic Settings](#basic-settings)
    - [Install SublimeText](#install-sublimetext)
        - [Easy GitLab or GitHub math: Add paired $ signs to the keybinds](#easy-gitlab-or-github-math-add-paired--signs-to-the-keybinds)
    - [Easily transform 2 spaced indent to 4 spaced indent](#easily-transform-2-spaced-indent-to-4-spaced-indent)
- [Install Git Bash](#install-git-bash)
    - [Install zstd and rsync for Git Bash](#install-zstd-and-rsync-for-git-bash)
- [Setup Git](#setup-git)
    - [Check your branches in git log history in a pretty line](#check-your-branches-in-git-log-history-in-a-pretty-line)
    - [Push with tags: multi-line git alias](#push-with-tags-multi-line-git-alias)
    - [GitLab Markdown math expressions for README.md, etc.](#gitlab-markdown-math-expressions-for-readmemd-etc)
    - [GitHub Markdown math expressions for README.md, etc.](#github-markdown-math-expressions-for-readmemd-etc)
    - [Git remote origin for SSH](#git-remote-origin-for-ssh)
    - [Make a new Git \(LFS\) repository from local](#make-a-new-git-lfs-repository-from-local)
    - [Manage multiple GitHub or GitLab accounts](#manage-multiple-github-or-gitlab-accounts)
        - [WSL and Windows shared ssh keys](#wsl-and-windows-shared-ssh-keys)
- [Install Docker Desktop for Windows](#install-docker-desktop-for-windows)
- [Install Python versions with pyenv-win and virtual environments with poetry](#install-python-versions-with-pyenv-win-and-virtual-environments-with-poetry)
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
- [Shell Scripting for convenience](#shell-scripting-for-convenience)
    - [Basic flag setup with getopts](#basic-flag-setup-with-getopts)
    - [Argparse-bash by nhoffman](#argparse-bash-by-nhoffman)

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

<a id="linux-development-guide"></a>
### Linux development guide

If using WSL fully without any interaction from Windows, follow the guide below.

Sometimes, however, using GitBash for committing or other functions that interact with the computer such as in audio libraries (pyaudio) it might be necessary to use both setups simultaneously.

https://github.com/elisa-aleman/linux-data-science-py-setup

<a id="wsl-about-dns-issues-under-vpn-or-proxy"></a>
### WSL: About DNS issues under VPN or proxy

Commands such as `sudo apt update` or `pip install ...` will sometimes fail under WSL and a VPN or office network.
This can also cause issues with docker, or any other development environment, so we need to confirm the following.

`/etc/resolv.conf` needs to have this part not commented. (Alternatively, it could also be added in `/etc/wsl.conf` )

```
[network]
generateResolvConf = false
```

Then, using `ipconfig|findstr DNS` to find your base domain, you can add it as well to `/etc/resolv.conf`:

```
search <YOUR_DOMAIN>
nameserver 8.8.8.8
nameserver 1.1.1.1
```


<a id="basic-settings"></a>
## Basic Settings

There are some basic things you should do to make programming in Windows easier.

- Setup the initial WiFi connection.
- When leaving the desk always press WIN+L to lock under password.
- For ease of use, make hidden files and file extensions visible.
    - Open an explorer window, under View > Show > File name extensions, Hidden items


<a id="install-git-bash"></a>
## Install Git Bash

I followed this guide for using git on Windows:<br>
https://www.pluralsight.com/guides/using-git-and-github-on-windows

Download and Install from: <br>
https://git-scm.com/download/win

I used most of the suggestions on installation, except for a few exceptions. Here's the options I chose:

- Check all the check-marks at the beginning to have right click menus for GitBash and shortcuts in the Windows pane.
- Make sure to install GitLFS along with the installation.
- Associate `.sh` and `.git` files with GitBash
- Use **Nano** editor as default
- Override the `master` branch to `main`
- When prompted to the **Adjusting your PATH environment window** : choose **Use Git from Git Bash only**
- Use Bundled OpenSSH
- Use OpenSSL
- When choosing line-endings: **Checkout as-is, commit Unix-style line endings**
- Use MinTTY
- Choose the default behavior of `git pull` as `fast-forward only`
- Use Git credential manager
- Enable file system caching
- No experimental tools with known bugs.

Now click **Install** and wait for the installation to finish.

One thing I noticed is that if I open the GitBash software from the Windows pane, windows command prompt `cmd` style commands such as `cls` will run, while if I right click a folder and run GitBash from that location, pseudo-Unix style commands such as `clear` will run. It is confusing and I'm used to Unix style commands so remember to always right click open GitBash.

<a id="install-zstd-and-rsync-for-git-bash"></a>
### Install zstd and rsync for Git Bash

Installing rsync is necessary to sync files with a Linux server from a windows machine, which is useful to run big programs like machine learning when that server has a powerful GPU. However, rsync is a Unix command so we have to install it manually to be able to use it with Git Bash.

https://shchae7.medium.com/how-to-use-rsync-on-git-bash-6c6bba6a03ca

1. Open Git Bash as administrator

2. Install zstd

```
mkdir zstd
cd zstd
curl -L -O https://github.com/facebook/zstd/releases/download/v1.4.4/zstd-v1.4.4-win64.zip
unzip zstd-v1.4.4-win64.zip
echo "alias zstd='~/zstd/zstd.exe'" >> ~/.bashrc
source ~/.bashrc
```

3. Get rsync.exe

```
cd
mkdir rsync
cd rsync
curl -L -O https://repo.msys2.org/msys/x86_64/rsync-3.2.3-1-x86_64.pkg.tar.zst
zstd -d rsync-3.2.3–1-x86_64.pkg.tar.zst
tar -xvf rsync-3.2.3-1-x86_64.pkg.tar
cp ./usr/bin/rsync.exe /usr/bin
```

4. Install dependencies

```
cd
mkdir libzstd
cd libzstd
curl -L -O https://repo.msys2.org/msys/x86_64/libzstd-1.4.8-1-x86_64.pkg.tar.zst
zstd -d libzstd-1.4.8-1-x86_64.pkg.tar.zst
tar -xvf libzstd-1.4.8-1-x86_64.pkg.tar
cp ./usr/bin/msys-zstd-1.dll /usr/bin
```

```
cd
mkdir libxxhash
cd libxxhash
curl -L -O https://repo.msys2.org/msys/x86_64/libxxhash-0.8.0-1-x86_64.pkg.tar.zst
zstd -d libxxhash-0.8.0-1-x86_64.pkg.tar.zst
tar -xvf libxxhash-0.8.0-1-x86_64.pkg.tar
cp ./usr/bin/msys-xxhash-0.8.0.dll /usr/bin
```

5. Check if rsync is working

On a normal git bash session:

```
rsync
```

It should output the different options you can use with it.


## Install Docker Desktop for Windows

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/desktop/install/windows-install/

Reboot after installing.

Running docker on WSL is also possible while having the Docker Desktop app open.
The desktop app needs to be running before being able to run commands on the shell (WSL recommended).

Test:`docker container --list`

---

That is all for now. This is my initial setup for the development environment. If I have any projects that need further tinkering, that goes on another repository / tutorial.

