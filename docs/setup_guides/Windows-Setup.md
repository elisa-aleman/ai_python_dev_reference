# Windows setup environment for development

TODO:

- [ ] propagate relevant parts of suggested tools guide to OS specific guides with links to this main document

I initially used to work on python and Data Science using MacOSX or Ubuntu, so changing to Windows was a confusing experience. Because of this I wrote a guide of everything I did for the initial setup of my computer to start programming.

This is how I set up a fresh windows PC to start working in machine learning and programming. I also keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.

My current Windows PC is using Windows 365.

## Install Visual Studio (or rather don't uninstall it)

Even though I don't use Visual Studio to program, I ran into some issues when I tried to uninstall it to avoid wasting space. It turns out that many applications use the executable code it is packaged with to interpret C++ programs. Without it I could not use `winget` to download applications or even open up the Notepad app.

Install it from the Microsoft Store if you ever get [this error](https://answers.microsoft.com/en-us/windows/forum/all/bad-image-msvcp140dll-error/85d957df-5985-451e-be2d-135e4b0c754e):

```
Windows Warning: Bad Image – MSVCP140.dll not designed to run on Windows.
```

As in the forum answers, I tried to install just the `Visual C++ Redistributable for Visual Studio 2015` software, but the problem was only fixed by installing the full Visual Studio program.


## Basic Settings

There are some basic things you should do to make programming in Windows easier.

- Setup the initial WiFi connection.
- When leaving the desk always press WIN+L to lock under password.
- For ease of use, make hidden files and file extensions visible.
    - Open an explorer window, under View > Show > File name extensions, Hidden items


## Install PowerToys, Key re-mapping

[PowerToys](https://github.com/Microsoft/powertoys/releases) is a Windows 10 or 11 collection of advanced settings which will come in handy. For example, I needed one to re-map the keys on my keyboard.


Personally, since my keyboard wasn't Japanese but I live in Japan and pretty much it's the layout I've gotten used to, I remapped:

```
]}           ->   \|
Alt(right)   ->   \_
Alt(left)+[{ ->   ]}
Shift+0      ->   )
Shift+9      ->   (
```

As I pretty much never use the right Alt key for anything I can't use the left one for, and I keep missing the parentheses because of the actual key cap being one number over.

## Terminal

Refer to my section in [Suggested Tools and Setup: Windows Terminal with Cmder, Git Bash, and WSL profiles](./Suggested-Tools-and-Setup.md#windows-terminal-with-cmder-git-bash-and-wsl-profiles)

## WSL (Windows Subsystem for Linux) installation guide

Reference:
- [How to install Windows Subsystem for Linux in Windows 11](https://www.groovypost.com/howto/install-windows-subsystem-for-linux-in-windows-11/)


1. Open the cmd with administrator privileges
2. `wsl --install`
3. Restart computer
4. Make username under new Linux terminal

### WSL: Paths to directories outside of the Linux environment

If in the above tutorial for separate git accounts, for example, you needed to use paths to locations in the Windows system, you can replace `C:` with `/mnt/c/`

### WSL: About .profile and .bash_profile

There is a difference between running an interactive shell inside of a started up Linux system, say, if you had Ubuntu installed the regular way, and running the main WSL window. 

Running the WSL software inside Windows opens a login shell, which is different from the interactive shell we are used to.

Login shells load `.profile`, which then reads `.bashrc` if the shell being used is bash. However, `.profile` is ignored if there exists a `.bash_profile`, which means usually that `.bashrc` will never be read, and so will also `.bash_aliases` not be read.

This can be fixed in a few ways:

1. Run the command `bash` every time at start up to open an interactive shell inside the login shell. The only difference this makes is that to exit WSL via commands you'd have to run `exit` on the interactive shell and then on the login shell, twice.

2. Move all the contents of `.bash_profile` to `.profile`, then delete `.bash_profile` so that there's nothing stopping all the initial codes from running even in a login shell.

3. Add `source ~/.profile` to the beginning of `.bash_profile` so that it is run regardless, and therefore also loads `.bashrc` if necessary. Personally I chose this one.

However, the best option is to actually start the WSL software with an initial command that starts bash at startup.

```cmd
:: cmd

C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l
```

This way, there's no need to edit the profile files.

Personally, I [added this command to the Windows Terminal profile as explained in "Suggested Tools and Setup: Windows Terminal with Cmder, Git Bash, and WSL profiles"](./Suggested-Tools-and-Setup.md#windows-terminal-with-cmder-git-bash-and-wsl-profiles) for WSL in the `Command line` setting.

### Linux development guide

If using WSL fully without any interaction from Windows, follow the [Linux / WSL Setup guide](./Linux-WSL-Setup.md) too.

Sometimes, however, using GitBash for committing or other functions that interact with the computer such as in audio libraries (`pyaudio`) it might be necessary to use both setups simultaneously.

This is because WSL interacts poorly with audio-visual devices that are technically available on the Windows side.

Moreover, to run docker in WSL, it has to be installed in Windows using Docker Desktop.

### WSL: About DNS issues under VPN or proxy

Commands such as `sudo apt update` or `pip install ...` will sometimes fail under WSL and a VPN or office network.
This can also cause issues with docker, or any other development environment, so we need to confirm the following.

`/etc/resolv.conf` needs to have this part not commented. (Alternatively, it could also be added in `/etc/wsl.conf` )

```
# /etc/resolv.conf

[network]
generateResolvConf = false
```

Then, using `ipconfig|findstr DNS` to find your base domain, you can add it as well to `/etc/resolv.conf`:

```
# /etc/resolv.conf

search <YOUR_DOMAIN>
nameserver 8.8.8.8
nameserver 1.1.1.1
```

## Setup git

Follow the [Git Setup and Customization](./Git-Setup-and-Customization.md) for more details.

## Install Docker Desktop for Windows

Docker allows us to run server apps that share an internal environment separate from the OS. This has the benefit of providing, sans any hardware incompatibilities, the same environment for anyone that uses the same docker image, leading to fewer compatibility issues when sharing our programs with others to run.

Follow [this official guide for docker](https://docs.docker.com/desktop/install/windows-install/)

Reboot after installing.

Running docker on WSL is also possible while having the Docker Desktop app open.
The desktop app needs to be running before being able to run commands on the shell (WSL recommended).

Test:`docker container --list`

The images built and pulled will be stored in:

```
%USERPROFILE%\AppData\Local\Docker\wsl
```

Which is on the C drive.

If you have a concern about the memory space you can change it in the settings:

1. Stop the Docker engine and quit any open WSL instances
2. Docker Desktop app settings
3. Resources
4. Advanced
5. Disk Image Location
6. Change it and click Apply & Restart

### Install zstd and rsync for Git Bash

Installing rsync is necessary to sync files with a Linux server from a windows machine, which is useful to run big programs like machine learning when that server has a powerful GPU. However, rsync is a Unix command so we have to install it manually to be able to use it with Git Bash.

References:

- [How to use rsync on git-bash](https://shchae7.medium.com/how-to-use-rsync-on-git-bash-6c6bba6a03ca)

1. Open Git Bash as administrator

2. Install zstd

```sh
# @ git-bash::~

mkdir zstd
cd zstd
curl -L -O https://github.com/facebook/zstd/releases/download/v1.4.4/zstd-v1.4.4-win64.zip
unzip zstd-v1.4.4-win64.zip
echo "alias zstd='~/zstd/zstd.exe'" >> ~/.bashrc
source ~/.bashrc
cd
```

3. Get rsync.exe

```sh
# @ git-bash::~

mkdir rsync
cd rsync
curl -L -O https://repo.msys2.org/msys/x86_64/rsync-3.2.3-1-x86_64.pkg.tar.zst
zstd -d rsync-3.2.3–1-x86_64.pkg.tar.zst
tar -xvf rsync-3.2.3-1-x86_64.pkg.tar
cp ./usr/bin/rsync.exe /usr/bin
cd
```

4. Install dependencies

```sh
# @ git-bash::~

mkdir libzstd
cd libzstd
curl -L -O https://repo.msys2.org/msys/x86_64/libzstd-1.4.8-1-x86_64.pkg.tar.zst
zstd -d libzstd-1.4.8-1-x86_64.pkg.tar.zst
tar -xvf libzstd-1.4.8-1-x86_64.pkg.tar
cp ./usr/bin/msys-zstd-1.dll /usr/bin
cd
```

```sh
# @ git-bash::~

mkdir libxxhash
cd libxxhash
curl -L -O https://repo.msys2.org/msys/x86_64/libxxhash-0.8.0-1-x86_64.pkg.tar.zst
zstd -d libxxhash-0.8.0-1-x86_64.pkg.tar.zst
tar -xvf libxxhash-0.8.0-1-x86_64.pkg.tar
cp ./usr/bin/msys-xxhash-0.8.0.dll /usr/bin
cd
```

5. Check if rsync is working

On a normal git bash session:

```sh
# @ git-bash::~
rsync
```

It should output the different options you can use with it.

## Install Python

Follow my [Python setup guide](./Python-Setup.md)

## CUDA and GPU settings

For accelerated GPU processing (specifically nvidia GPUs) for your AI programs I recommend using WSL and a Docker image that uses [CUDA](https://en.wikipedia.org/wiki/CUDA), as described in my [Linux WSL Setup: CUDA and GPU settings guide](./Linux-WSL-Setup.md#CUDA-and-GPU-settings).

For specific examples where I used CUDA compatible images, see my [CUDA python dockerfiles document](../ai_development/CUDA-python-dockerfiles.md)

## Avoid updates while running programs

References:
- [Windows Updates Restart Disable](https://www.thewindowsclub.com/windows-updates-restart-disable)

---

That is all for now. This is my initial setup for the development environment. If I have any projects that need further tinkering, that goes on another repository / tutorial.

