
# Mac Python setup environment for development (on a proxy)

TODO:
- [ ] Re-check old narrative
- [ ] Check all syntax highlights and location comments
- [ ] Add style guide for location comments
- [ ] Wrap all links
- [ ] propagate relevant parts of suggested tools guide to OS specific guides with links to this main document


This is how I set up a fresh mac to start working in machine learning and programming. My lab runs under a proxy, so all the settings have that one extra ...sigh... task to work correctly. I keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.


## Basic Settings

Setup root password:
https://www.cyberciti.biz/faq/how-to-change-root-password-on-macos-unix-using-terminal/

```sh
# @ shell(mac_osx)

sudo passwd root
```

- Set up the screen lock command so you can do it every time you stand up:
    - https://achekulaev.medium.com/lock-screen-with-cmd-l-l-global-hotkey-on-macos-high-sierra-3c596b76026a
    - Settings>Keyboard>Keyboard Shortcuts>App Shortcuts
        + All Applications
            * `Lock Screen`
            * `CMD+L`

- Set up your WiFi connection.
- For ease of use, make hidden files and file extensions visible.

Hidden files visibility:
I can't work without seeing hidden files, so in the newer versions of MacOSX we can do `CMD+Shift+.` and the hidden files will appear. 
If this doesn't work, open up the terminal and:

```sh
# @ shell(mac_osx)

defaults write com.apple.finder AppleShowAllFiles YES
```



## Setup proxy system wise


### Normal settings

First, know the {PROXY_HOST} url and the {PORT} that you need to access in your specific place of work. Then put those in the system settings as appropriate.

`Settings > Network > Advanced > Proxies`

Web Proxy (HTTP)
`{PROXY_HOST}:{PORT}`

Secure Proxy (HTTPS)
`{PROXY_HOST}:{PORT}`

FTP Proxy
`{PROXY_HOST}:{PORT}`

Bypass proxy settings for these Hosts & Domains:
`*.local, 169.254/16, 127.0.0.1, HOST_URL`

Depending on your organization, setting a HOST_URL to bypass here might also be necessary. Check with your administrator.


### Time settings

Some proxies need you to set up the time of the computer to match the network in order to work correctly. Mine does at least. So the settings are such:

Settings > Date & Time > Date & Time

Set Date and Time automatically:
{TIME_URL}

The {TIME_URL} will depend on your organization, so check with your administrator.


## Setup proxy settings in bash

So now that the system settings are out of the way, we need to setup the proxy addresses in bash so that programs we run take those variables, since the system setup doesn't reach deep enough for several tools we use.

So we will add these settings to `.zprofile` to load each time we login to our user session. 

```sh
# @ shell(mac_osx)

sudo nano ~/.zprofile
```

This will open a text editor inside the terminal. If the file is new it will be empty.
Add the following lines: (copy and paste)

```sh
# @ nano::~/.zprofile

export http_proxy={PROXY_HOST}:{PORT}
export https_proxy={PROXY_HOST}:{PORT}
export all_proxy={PROXY_HOST}:{PORT}
export HTTP_PROXY={PROXY_HOST}:{PORT}
export HTTPS_PROXY={PROXY_HOST}:{PORT}
export ALL_PROXY={PROXY_HOST}:{PORT}
export no_proxy=localhost,127.0.0.1,169.254/16,HOST_URL
export NO_PROXY=localhost,127.0.0.1,169.254/16,HOST_URL

# CTRL+O
# CTRL+X
```

Then these ones for logging in ssh to the servers in the laboratory without typing it every time.

```sh
# @ shell(mac_osx)
alias lab_server="ssh -p {PORT} {USERNAME}@{HOST_IP_ADDRESS}"
```

Of course, you'll need your own {PORT} and {USERNAME} and {HOST_IP_ADDRESS} here, depending on where you want to log in.

Press `CTRL+O` to write, press `ENTER` to keep the name, then press `CTRL+X` to close the editor

Relaunch the terminal.


## Install Homebrew

For more info, click [here](https://brew.sh).

First we need to consider the macOS Requirements from their website:

- A 64-bit Intel CPU
- macOS High Sierra (10.13) (or higher)
- Command Line Tools (CLT) for Xcode: `xcode-select --install`, [developer.apple.com/downloads](https://developer.apple.com/downloads) or [Xcode](https://itunes.apple.com/us/app/xcode/id497799835)
- A Bourne-compatible shell for installation (e.g. bash or zsh)

As it says in the third requirement, we need the Command Line Tools for Xcode.


### Install Xcode Command Line Tools

We have 3 options:

- Install just the tools:

```sh
# @ shell(mac_osx)

xcode-select --install
```

- Download them from the official Apple website

Go to [developer.apple.com/downloads](https://developer.apple.com/downloads) and sign in with your Apple ID and password.

Agree to the Apple Developer Agreement.

Select the latest non-beta Command Line Tools for Xcode available, download de .dmg file and install.

- Install the full Xcode app

Xcode as an app is really heavy, so if you don't intend to work directly on the IDE of Xcode or on any other uses of the app, I don't recommend it. I also have no experience with setting up the CLT with this path.

For this option, you also need to sign up to be an Apple Developer. 


### Install Homebrew with no proxy

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`


### Install Homebrew under a proxy

Now that we have the CLT, we can proceed.

First configure your git proxy settings:
```sh
# @ shell(mac_osx)

git config --global http.proxy http://{PROXY_HOST}:{PORT}
```
Replace your {PROXY_HOST} and your {PORT}.

Then install homebrew using proxy settings as well:
```sh
# @ shell(mac_osx)

/bin/bash -c "$(curl -x {PROXY_HOST}:{PORT} -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

And finally alias `brew` so it always uses your proxy settings:

```sh
# @ shell(mac_osx)

alias brew="https_proxy={PROXY_HOST}:{PORT} brew"
```

Otherwise, if you're not under a proxy just follow the instructions here:
https://docs.brew.sh/Installation


## Curl proxy settings

Right there installing Homebrew we used explicit proxy settings on the curl command to avoid any issues, but to avoid doing this every time for future uses of curl, we also need to setup proxy settings.

```sh
# @ shell(mac_osx)

sudo nano ~/.curlrc
```

And add the following:
```sh
# @ nano::~/.curlrc

proxy = {PROXY_HOST}:{PORT} 
```



## Install Docker Desktop for Mac

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/desktop/install/mac-install/

- Use Apple menu >About this Mac. 
- Confirm the Chip to determine the installation method

The desktop app needs to be running before being able to run commands on the terminal.

Test:`docker container --list`


---

That is all for now. This is my initial setup for the lab environment under a proxy. If I have any projects that need further tinkering, that goes on another repository / tutorial.

