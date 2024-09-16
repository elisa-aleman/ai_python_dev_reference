# Windows setup environment for development

TODO:
- [ ] Re-check old narrative
- [ ] Add comments about proxy?
- [ ] Check all syntax highlights and location comments
- [ ] Add style guide for location comments
- [ ] Wrap all links
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

When developing, most of your time will be spent on either your text editor / IDE, or your terminal executing code. It is therefore important to choose a good one that integrates well into your workflow.

Personally, when I started development on Ubuntu about 15 years ago, I wasn't too aware of this and the default gnome terminal was enough for me, although I did customize it a little. Then, when I moved over to MacOSX, my goal was to make my development environment as close to Ubuntu as possible. It was then that I started finding better options since the default terminal in MacOSX had zero customization and the color theme was awful on my eyes. Once finding better options, I found myself using features that I never used in Linux. Finally, when I moved to Windows because of my job requirements and also at home, for gaming, I found the OS to be truly disgusting, and the native development tools to be absolutely horrid. However, most of that stopped mattering when I found [WSL (Windows Subsystem for Linux)](https://ubuntu.com/desktop/wsl), I could run all my commands within a contained Linux environment, and all that I had to do on the Windows side was have my editor open (which is cross-platform). Then, on my free time, I could game without issues or partitioning memory drives, etc. (Although Windows 11 is frustrating enough that this might change).

I quickly realized that I needed a good Terminal setup to approximate the workflow I had on MacOSX.

Here are my recommendations:

### Windows Terminal with Cmder, Git Bash, and WSL profiles

It comes pre-installed, but compared to the pre-installed MacOSX terminal, it pretty much has all the functions that you need for customizing it to your taste, and the capability to run behind the scenes with better software than it initially has (looking at you, PowerShell). It also is not [Electron](https://www.electronjs.org/apps) based, which makes it **lightweight and fast** in comparison to other bloated options based in Electron. Don't use Electron for things where speed is paramount, please.

I will split this in three sections:

1) [Cmder](https://cmder.app/): Having an actually usable command prompt for Windows commands combined with a few essential sh commands.
2) [Git Bash](https://gitforwindows.org/): Having a local (still technically within windows OS) bash shell for every other command you need.
3) [WSL](https://www.groovypost.com/howto/install-windows-subsystem-for-linux-in-windows-11/): Access the Windows Subsystem for Linux within the Windows Terminal correctly

#### Windows Terminal + Cmder

[Cmder](https://cmder.app/) is a powerful combination of `cmd`, Clink, ConEmu and Git Bash that actually makes working with Windows commands bearable. However, I prefer to use the Windows Terminal as an interface. We can combine both by assigning a new profile to Windows Terminal pointing to the installed Cmder.

[Here's a pretty thorough guide for that](https://medium.com/talpor/windows-terminal-cmder-%EF%B8%8F-573e6890d143)

Besides the manual starting directory `D:\git\` I personalized, the guide is pretty complete.

#### Windows Terminal + Git Bash

Download and Install from [The Git for Windows website](https://gitforwindows.org/)

I used most of the suggestions on installation, except for a few exceptions. Here's the options I chose:

- Check all the check-marks at the beginning to have right click menus for GitBash and shortcuts in the Windows pane.
- Make sure to install GitLFS along with the installation.
- Associate `.sh` and `.git` files with GitBash
- Use **Nano** editor as default
- Override the `master` branch to `main`
- When prompted to the **Adjusting your PATH environment window** : choose **Use Git from the command line and also 3rd party software**
- Use Bundled OpenSSH
- Use OpenSSL
- When choosing line-endings: **Checkout windows, commit Unix-style line endings**
- Use MinTTY
- Choose the default behavior of `git pull` as `fast-forward only`
- Do not use Git credential manager (later I will add SSH configs)
- Enable file system caching
- Enable pseudo consoles

Now click **Install** and wait for the installation to finish.

Follow the [Git Setup and Customization](./Git-Setup-and-Customization.md) for more.

What's also good about this setup is that it will add a Windows Terminal profile.

Then, I setup my starting directory `D:\git\`

#### Windows Terminal + WSL

Reference:
- [How to install Windows Subsystem for Linux in Windows 11](https://www.groovypost.com/howto/install-windows-subsystem-for-linux-in-windows-11/)


1. Open the cmd with administrator privileges
2. `wsl --install`
3. Restart computer
4. Make username under new Linux terminal

Then, setup a new profile for WSL on Windows Terminal:

1) Settings
2) Add a new profile
3) Name: `Ubuntu`
4) Command line `C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l`
5) Starting directory: `/mnt/d/git/`
6) Icon: `https://assets.ubuntu.com/v1/49a1a858-favicon-32x32.png`

And it should work!

So, there's a few points to clarify:

- Paths in WSL referring to the Windows disks can be found under `/mnt/` as mounted volumes.
- Make sure it's the `C:\Windows\system32\wsl.exe -d Ubuntu` and not the `ubuntu.exe`, because they will behave differently under Windows Terminal
- The original command `C:\Windows\system32\wsl.exe -d Ubuntu` does not open an additional bash, [leaving you in a login shell instead of an interactive shell](./Windows-Setup.md#wsl-about-profile-and-bash_profile). However, we can change the initial command that starts bash at startup to get the interactive shell automatically.

```cmd
C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l
```

This way, there's no need to edit the profile files.


Also, do refer to my [Windows Setup guide: WSL section](./Windows-Setup.md#wsl-windows-subsystem-for-linux-installation-guide), since there are a few peculiarities in proxy environments.

#### Windows Terminal + SSH server

When developing, it will be common that the main work is all done over a Linux server with a GPU, and that the main Windows PC is just used for editing the files.

To access the SSH server over a new profile, we can run the startup command like this:

```cmd
C:/Program Files/Git/bin/bash.exe -i -l -c 'ssh USER@HOST:PORT'
```

The host can also be added to the `.ssh/config` file:

```
@ ~/.ssh/config 
Host linux_server
    HostName xxx.xx.x.xx
    Port xxx
    User xxxx
    RequestTTY force
    RemoteCommand cd your/path/here; bash -l

```

Then, the command can be:

```cmd
C:/Program Files/Git/bin/bash.exe -i -l -c 'ssh USER@linux_server'
```


#### Windows Terminal themes

I found this pretty comprehensive free [Windows Terminal themes collection](https://windowsterminalthemes.dev/).

To implement a theme, all you have to do is copy the JSON code, then open Windows Terminal, go under Settings, then the lower left corner must have a `Open JSON file` button, which points to the actual Windows Terminal configuration file, where you can paste the themes where appropriate (alongside other themes, and remember to check that the commas match)


## WSL (Windows Subsystem for Linux) installation guide

Reference:
- [How to install Windows Subsystem for Linux in Windows 11](https://www.groovypost.com/howto/install-windows-subsystem-for-linux-in-windows-11/)


1. Open the cmd with administrator privileges
2. `wsl --install`
3. Restart computer
4. Make username under new Linux terminal

### WSL: Paths to directories outside of the Linux environment

If in the above tutorial for separate git accounts, for example, you needed to use paths to locations in the Windows system, you can replace C: with /mnt/c/

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
C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l
```

This way, there's no need to edit the profile files.

### Linux development guide

If using WSL fully without any interaction from Windows, follow the [Linux / WSL Setup guide](./Linux-WSL-Setup.md) too.

Sometimes, however, using GitBash for committing or other functions that interact with the computer such as in audio libraries (`pyaudio`) it might be necessary to use both setups simultaneously.

This is because WSL interacts poorly with audio-visual devices that are technically available on the Windows side.

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




## Install Docker Desktop for Windows

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/desktop/install/windows-install/

Reboot after installing.

Running docker on WSL is also possible while having the Docker Desktop app open.
The desktop app needs to be running before being able to run commands on the shell (WSL recommended).

Test:`docker container --list`


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


---

That is all for now. This is my initial setup for the development environment. If I have any projects that need further tinkering, that goes on another repository / tutorial.

