# Mac Python setup environment for Machine learning Laboratory on a proxy

This is how I set up a fresh mac to start working in machine learning and programming. My lab runs under a proxy, so all the settings have that one extra ...sigh... task to work correctly. I keep this tutorial handy in case I do a clean OS install or if I need to check some of my initial settings.

<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [Basic Settings](#basic-settings)
    - [Install SublimeText](#install-sublimetext)
        - [Easy TeX math: Add paired $ signs to the keybinds](#easy-tex-math-add-paired--signs-to-the-keybinds)
    - [Easily transform 2 spaced indent to 4 spaced indent](#easily-transform-2-spaced-indent-to-4-spaced-indent)
- [Setup proxy system wise](#setup-proxy-system-wise)
    - [Normal settings](#normal-settings)
    - [Time settings](#time-settings)
- [Setup proxy settings in bash](#setup-proxy-settings-in-bash)
- [Install Homebrew](#install-homebrew)
    - [Install Xcode Command Line Tools](#install-xcode-command-line-tools)
    - [Install Homebrew with no proxy](#install-homebrew-with-no-proxy)
    - [Install Homebrew under a proxy](#install-homebrew-under-a-proxy)
- [Curl proxy settings](#curl-proxy-settings)
- [Install and setup Git](#install-and-setup-git)
    - [Check your branches in git log history in a pretty line](#check-your-branches-in-git-log-history-in-a-pretty-line)
    - [Push with tags: multi-line git alias](#push-with-tags-multi-line-git-alias)
    - [GitHub Markdown math expressions for README.md, etc.](#github-markdown-math-expressions-for-readmemd-etc)
    - [GitLab Markdown math expressions for README.md, etc.](#gitlab-markdown-math-expressions-for-readmemd-etc)
    - [Install Git Large File System](#install-git-large-file-system)
    - [Make a new Git \(LFS\) repository from local](#make-a-new-git-lfs-repository-from-local)
    - [Manage multiple GitHub or GitLab accounts](#manage-multiple-github-or-gitlab-accounts)
- [Install Docker Desktop for Mac](#install-docker-desktop-for-mac)
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
- [Shell Scripting for convenience](#shell-scripting-for-convenience)
    - [Basic flag setup with getopts](#basic-flag-setup-with-getopts)
    - [Argparse-bash by nhoffman](#argparse-bash-by-nhoffman)

<!-- /MarkdownTOC -->


<a id="basic-settings"></a>
## Basic Settings

Setup root password:
https://www.cyberciti.biz/faq/how-to-change-root-password-on-macos-unix-using-terminal/

```
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
```
defaults write com.apple.finder AppleShowAllFiles YES
```


<a id="install-sublimetext"></a>
### Install SublimeText

- Install SublimeText4 for ease of use (this is my personal favorite, but it's not necessary)

https://www.sublimetext.com/download

- Paste the SublimeText4 preferences (my personal preferences)

```
{
    "ignored_packages":
    [
        "Vintage",
    ],
    "spell_check": true,
    "tab_size": 4,
    "translate_tabs_to_spaces": true,
    "copy_with_empty_selection": false
}
```

Also, Sublime Text is all about the plugins. Install Package Control by typing CTRL+Shift+P, then typing "Install Package Control"

Then here's some cool packages to try:

- [LaTeXTools](https://packagecontrol.io/packages/LaTeXTools)
- [MarkdownTOC](https://packagecontrol.io/packages/MarkdownTOC)
- [MarkdownPreview](https://packagecontrol.io/packages/MarkdownPreview)
- [MarkdownEditing](https://packagecontrol.io/packages/MarkdownEditing)
- [Alignment](https://packagecontrol.io/packages/Alignment)
- [IncrementSelection](https://packagecontrol.io/packages/Increment%20Selection)
- [Selection Evaluator](https://packagecontrol.io/packages/Selection%20Evaluator)
- [Paste as One Line](https://packagecontrol.io/packages/Paste%20as%20One%20Line)
- [Invert Current Color Scheme](https://packagecontrol.io/packages/Invert%20Current%20Color%20Scheme)
- [PackageResourceViewer](https://packagecontrol.io/packages/PackageResourceViewer)

Now, for the Invert Current Color Scheme, I have my own fork that works with Sublime Text 4, so use the PackageResourceViewer to replace the main python file with my code:
 
https://github.com/elisa-aleman/sublime-invert-current-color-scheme

In MarkdownTOC.sublime-settings, paste the following for hyperlink markdowns and compatibility with MarkdownPreview:

```
{
  "defaults": {
    "autoanchor": true,
    "autolink": true,
    "markdown_preview": "github",
    "uri_encoding": false
  },
}
```

After installing Markdown Editing, add this to the SublimeText4 preferences (my personal preferences)

```
"mde.auto_fold_link.enabled": false,
```

<a id="easy-tex-math-add-paired--signs-to-the-keybinds"></a>
#### Easy TeX math: Add paired $ signs to the keybinds

I found myself needing paired $dollar signs$ for math expressions in either LaTeX, GitLab with KeTeX or GitHub also with a different syntax but still some interpretation of TeX.

Searching for [how to do it on macros](https://forum.sublimetext.com/t/snippet-wrap-current-line-or-selection/53285), I found this post about keybindings which is a way better solution:

https://stackoverflow.com/questions/34115090/sublime-text-2-trying-to-escape-the-dollar-sign

Which, as long as we implement the double escaped dollar sign solution, we can use freely.

- Preferences > Key Bindings:
- Add this inside the brackets:

```
[
    // Auto-pair dollar signs on TeX or LaTeX
    { "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$$0\\$"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "source.text.tex" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^(?:\t| |\\)|]|\\}|>|$)", "match_all": true },
            { "key": "preceding_text", "operator": "not_regex_contains", "operand": "[\\$a-zA-Z0-9_]$", "match_all": true },
            { "key": "eol_selector", "operator": "not_equal", "operand": "string.quoted.double", "match_all": true }
        ]
    },
    { "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$${0:$SELECTION}\\$"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "source.text.tex" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": false, "match_all": true }
        ]
    },
    { "keys": ["$"], "command": "move", "args": {"by": "characters", "forward": true}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "source.text.tex" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
        ]
    },
    { "keys": ["backspace"], "command": "run_macro_file", "args": {"file": "Packages/Default/Delete Left Right.sublime-macro"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "source.text.tex" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "preceding_text", "operator": "regex_contains", "operand": "\\$$", "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
        ]
    },
    // Auto-pair dollar signs on Markdown for github
    { "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$$0\\$"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "text.html.markdown" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^(?:\t| |\\)|]|\\}|>|$)", "match_all": true },
            { "key": "preceding_text", "operator": "not_regex_contains", "operand": "[\\$a-zA-Z0-9_]$", "match_all": true },
            { "key": "eol_selector", "operator": "not_equal", "operand": "string.quoted.double", "match_all": true }
        ]
    },
    { "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$${0:$SELECTION}\\$"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "text.html.markdown" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": false, "match_all": true }
        ]
    },
    { "keys": ["$"], "command": "move", "args": {"by": "characters", "forward": true}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "text.html.markdown" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
        ]
    },
    { "keys": ["backspace"], "command": "run_macro_file", "args": {"file": "Packages/Default/Delete Left Right.sublime-macro"}, "context":
        [
            { "key": "selector", "operator": "equal", "operand": "text.html.markdown" },
            { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
            { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
            { "key": "preceding_text", "operator": "regex_contains", "operand": "\\$$", "match_all": true },
            { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
        ]
    },
]

```

<a id="easily-transform-2-spaced-indent-to-4-spaced-indent"></a>
### Easily transform 2 spaced indent to 4 spaced indent

https://forum.sublimetext.com/t/can-i-easily-change-all-existing-2-space-indents-to-4-space-indents/40158/2

- Sublime text, lower right corner
- Click on Spaces
- Select the current space number
- Click Convert indentation to Tabs
- Select the desired space number
- Click Convert indentation to Spaces

<a id="setup-proxy-system-wise"></a>
## Setup proxy system wise

<a id="normal-settings"></a>
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

<a id="time-settings"></a>
### Time settings

Some proxies need you to set up the time of the computer to match the network in order to work correctly. Mine does at least. So the settings are such:

Settings > Date & Time > Date & Time

Set Date and Time automatically:
{TIME_URL}

The {TIME_URL} will depend on your organization, so check with your administrator.

<a id="setup-proxy-settings-in-bash"></a>
## Setup proxy settings in bash

So now that the system settings are out of the way, we need to setup the proxy addresses in bash so that programs we run take those variables, since the system setup doesn't reach deep enough for several tools we use.

So we will add these settings to `.zprofile` to load each time we login to our user session. 

```
sudo nano .zprofile
```

This will open a text editor inside the terminal. If the file is new it will be empty.
Add the following lines: (copy and paste)

```
export http_proxy={PROXY_HOST}:{PORT}
export https_proxy={PROXY_HOST}:{PORT}
export all_proxy={PROXY_HOST}:{PORT}
export HTTP_PROXY={PROXY_HOST}:{PORT}
export HTTPS_PROXY={PROXY_HOST}:{PORT}
export ALL_PROXY={PROXY_HOST}:{PORT}
export no_proxy=localhost,127.0.0.1,169.254/16,HOST_URL
export NO_PROXY=localhost,127.0.0.1,169.254/16,HOST_URL
```

Then these ones for logging in ssh to the servers in the laboratory without typing it every time.

```
alias lab_server="ssh -p {PORT} {USERNAME}@{HOST_IP_ADDRESS}"
```

Of course, you'll need your own {PORT} and {USERNAME} and {HOST_IP_ADDRESS} here, depending on where you want to log in.

Press `CTRL+O` to write, press `ENTER` to keep the name, then press `CTRL+X` to close the editor

Relaunch the terminal.

<a id="install-homebrew"></a>
## Install Homebrew

For more info, click [here](https://brew.sh).

First we need to consider the macOS Requirements from their website:

- A 64-bit Intel CPU
- macOS High Sierra (10.13) (or higher)
- Command Line Tools (CLT) for Xcode: `xcode-select --install`, [developer.apple.com/downloads](https://developer.apple.com/downloads) or [Xcode](https://itunes.apple.com/us/app/xcode/id497799835)
- A Bourne-compatible shell for installation (e.g. bash or zsh)

As it says in the third requirement, we need the Command Line Tools for Xcode.

<a id="install-xcode-command-line-tools"></a>
### Install Xcode Command Line Tools

We have 3 options:

- Install just the tools:

```
xcode-select --install
```

- Download them from the official Apple website

Go to [developer.apple.com/downloads](https://developer.apple.com/downloads) and sign in with your Apple ID and password.

Agree to the Apple Developer Agreement.

Select the latest non-beta Command Line Tools for Xcode available, download de .dmg file and install.

- Install the full Xcode app

Xcode as an app is really heavy, so if you don't intend to work directly on the IDE of Xcode or on any other uses of the app, I don't recommend it. I also have no experience with setting up the CLT with this path.

For this option, you also need to sign up to be an Apple Developer. 

<a id="install-homebrew-with-no-proxy"></a>
### Install Homebrew with no proxy

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

<a id="install-homebrew-under-a-proxy"></a>
### Install Homebrew under a proxy

Now that we have the CLT, we can proceed.

First configure your git proxy settings:
```
git config --global http.proxy http://{PROXY_HOST}:{PORT}
```
Replace your {PROXY_HOST} and your {PORT}.

Then install homebrew using proxy settings as well:
```
/bin/bash -c "$(curl -x {PROXY_HOST}:{PORT} -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

And finally alias `brew` so it always uses your proxy settings:

```
alias brew="https_proxy={PROXY_HOST}:{PORT} brew"
```

Otherwise, if you're not under a proxy just follow the instructions here:
https://docs.brew.sh/Installation

<a id="curl-proxy-settings"></a>
## Curl proxy settings

Right there installing Homebrew we used explicit proxy settings on the curl command to avoid any issues, but to avoid doing this every time for future uses of curl, we also need to setup proxy settings.

```
sudo nano ~/.curlrc
```

And add the following:
```
proxy = {PROXY_HOST}:{PORT} 
```

<a id="install-and-setup-git"></a>
## Install and setup Git

Mac already has git installed (that's how we installed homebrew too), but we can update it and manage it with homebrew.

```
brew install git
```

Then setup the configuration. Make an account at [GitHub](https://github.com) to get a username and email associated with Git. Type the settings on the terminal. My settings are like this:

```
git config --global http.proxy http://{PROXY_HOST}:{PORT}
git config --global user.name {YOUR_USERNAME}
git config --global user.email {YOUR_EMAIL}
git config --global color.ui auto
git config --global merge.conflictstyle diff3
git config --global core.editor nano
git config --global core.autocrlf input
git config --global core.fileMode false
git config --global pull.ff only
```

This should make a file `~/.gitconfig` with the following text
```
# ~/.gitconfig

[http]
    proxy = http://{PROXY_HOST}:{PORT}
[user]
    name = YOUR_USERNAME
    email = YOUR_EMAIL
[color]
    ui = auto
[merge]
    conflictstyle = diff3
[core]
    editor = nano
    autocrlf = input
    fileMode = false
[pull]
    ff = only
[alias]
    adog = log --all --decorate --oneline --graph
```

That last one, `git adog` is very useful as I explain in [Check your branches in git log history in a pretty line](#check-your-branches-in-git-log-history-in-a-pretty-line)

Now to make your Mac ignore those pesky Icon? files in git:

The best place for this is in your global gitignore configuration file. You can create this file, access it, and then edit per the following steps (we need to use vim because of a special character in the Icon? files):

```
git config --global core.excludesfile ~/.gitignore_global
vim ~/.gitignore_global
```

press `i` to enter insert mode
type `Icon` on a new line
while on the same line, ctrl + v, ENTER, ctrl + v, ENTER
Also, let's add `.DS_Store` to .gitignore_global as well.

press ESC, then type `:wq` then hit ENTER.

<a id="check-your-branches-in-git-log-history-in-a-pretty-line"></a>
### Check your branches in git log history in a pretty line

This makes your history tree pretty and easy to understand inside of the terminal.
I found this in https://stackoverflow.com/a/35075021

```
git log --all --decorate --oneline --graph
```

Not everyone would be doing a git log all the time, but when you need it just remember: 
"A Dog" = git log --all --decorate --oneline --graph

Actually, let's set an alias:

```
git config --global alias.adog "log --all --decorate --oneline --graph"
```

This adds the following to the .gitconfig file:

```
[alias]
        adog = log --all --decorate --oneline --graph
```

And you run it like:

```
git adog
```

<a id="push-with-tags-multi-line-git-alias"></a>
### Push with tags: multi-line git alias

To add a multi-line alias, for example, push and then push the tags on one single command, use `'!git ... && git ...'` as a format:

Push with tags:
`git config --global alias.pusht '!git push && git push --tags'`

<a id="github-markdown-math-expressions-for-readmemd-etc"></a>
### GitHub Markdown math expressions for README.md, etc.

Following this guide, math is different in GitLab markdown than say, GitHub or LaTeX.
However, inside of the delimiters, it renders it using KaTeX, which uses LaTeX math syntax! 

https://docs.gitlab.com/ee/user/markdown.html#math

Inline: 
```
> $a^2 + b^2 = c^2$
```

Renders as: $a^2 + b^2 = c^2$

Block:
```
> $$a^2 + b^2 = c^2$$
```

Renders as:

$$a^2 + b^2 = c^2$$

But it only supports one line of math, so for multiple lines you have to do this:

```
> $$a^2 + b^2 = c^2$$
> <!-- (line break is important) -->
> $$c = \sqrt{ a^2 + b^2 }$$
```

Renders as:

$$a^2 + b^2 = c^2$$

$$c = \sqrt{ a^2 + b^2 }$$

It can even display matrices and the like:

```
> $$
> l_1 = 
> \begin{bmatrix}
>     \begin{bmatrix}
>         x_1 & y_1
>     \end{bmatrix} \\
>     \begin{bmatrix}
>         x_2 & y_2
>     \end{bmatrix} \\
>     ... \\
>     \begin{bmatrix}
>         x_n & y_n
>     \end{bmatrix} \\
> \end{bmatrix}
> $$
```

$$
l_1 = 
\begin{bmatrix}
    \begin{bmatrix}
        x_1 & y_1
    \end{bmatrix} \\
    \begin{bmatrix}
        x_2 & y_2
    \end{bmatrix} \\
    ... \\
    \begin{bmatrix}
        x_n & y_n
    \end{bmatrix} \\
\end{bmatrix}
$$


However, % comments will break the environment.

Math syntax in LaTeX:

https://katex.org/docs/supported.html

<a id="gitlab-markdown-math-expressions-for-readmemd-etc"></a>
### GitLab Markdown math expressions for README.md, etc.

Following this guide, math is different in GitLab markdown than say, GitHub or LaTeX.
However, inside of the delimiters, it renders it using KaTeX, which uses LaTeX math syntax! 

https://docs.gitlab.com/ee/user/markdown.html#math

Inline: 
```
> $`a^2 + b^2 = c^2`$
```

Renders as: $`a^2 + b^2 = c^2`$

Block:
```
> ```math
> a^2 + b^2 = c^2
> ```
```

Renders as:

```math
a^2 + b^2 = c^2
```

But it only supports one line of math, so for multiple lines you have to do this:

```
> ```math
> a^2 + b^2 = c^2
> ```
> ```math
> c = \sqrt{ a^2 + b^2 }
> ```
```

Renders as:

```math
a^2 + b^2 = c^2
```
```math
c = \sqrt{ a^2 + b^2 }
```

It can even display matrices and the like:

```
> ```math
> l_1 = 
> \begin{bmatrix}
>     \begin{bmatrix}
>         x_1 & y_1
>     \end{bmatrix} \\
>     \begin{bmatrix}
>         x_2 & y_2
>     \end{bmatrix} \\
>     ... \\
>     \begin{bmatrix}
>         x_n & y_n
>     \end{bmatrix} \\
> \end{bmatrix}
> ```
```

```math
l_1 = 
\begin{bmatrix}
    \begin{bmatrix}
        x_1 & y_1
    \end{bmatrix} \\
    \begin{bmatrix}
        x_2 & y_2
    \end{bmatrix} \\
    ... \\
    \begin{bmatrix}
        x_n & y_n
    \end{bmatrix} \\
\end{bmatrix}
```

However, % comments will break the environment.

Math syntax in LaTeX:

https://katex.org/docs/supported.html

<a id="install-git-large-file-system"></a>
### Install Git Large File System

This is for files larger than 50 MB to be able to be used in Git. Still, GitLFS has some limitations if you don't buy data packages to increase your usage limit. By default you get 1GB of storage and 1GB of bandwidth (how much you push or pull per month). For 5$USD, you can add a *data pack* that adds 50GB bandwith and 50GB Git LFS storage.

Now we need to install the git-lfs package to use it:

```
brew install git-lfs
```

Now I personally had issues with git-lfs not pushing or pulling because of my proxy.

This was fixed by checking my exported variables in `.zprofile`.

The problem was it was set up like this:

```
export https_proxy=http://{PROXY_HOST}:{PORT}
export HTTPS_PROXY=https://{PROXY_HOST}:{PORT}
```

Where the proxy host at my lab doesn't manage the `https:// ` addresses correctly. So I had to correct them and remove the `s` like this:

```
export https_proxy=http://{PROXY_HOST}:{PORT}
export HTTPS_PROXY=http://{PROXY_HOST}:{PORT}
```

So the `https_proxy` variables still point to a `http://` address. It's not the best but in my network there was no other choice. 

<a id="make-a-new-git-lfs-repository-from-local"></a>
### Make a new Git (LFS) repository from local

Now that we have Git and Python installed, we can make our first project. I like to leave this part of the tutorial in even if it doesn't classify as a setup because using Git and GitLFS was confusing at first.

First make a repository on GitHub with no .gitignore, no README and no license.
Then, on local terminal, cd to the directory of your project and initialize git
```
cd path/to/your/project
git init
```

If using Git LFS:
```
git lfs install
```
It's supposed to be ready, but first, let's make a few hooks executable
```
chmod +x .git/hooks/*
```

Make a .gitignore depending on which files you don't want in the repository and add it
```
git add .gitignore
```

If using Git LFS, add the tracking settings for this project (For example, heavy csv files in this case)
```
git lfs track "*.csv"
```

And then add them to git
```
git add .gitattributes
```

Commit these changes first
```
git commit -m "First commit, add .gitignore and .gitattributes"
```

Now add all the data from your local repository. `git add .` adds all the files in the folder.
```
git add .
```

Depending on the size of your project, it might be wiser to add it in parts instead of all at once. e.g.
```
git add *.py
git add *.csv
...
```
or
```
git add dir1
git add dir2
...
```

Check if all the paths are added
```
git status
```

Check if all the Git LFS files are tracked correctly
```
git lfs ls-files
```

If so, commit.
```
git commit -m "First data commit"
```

Set the new remote URL from the repository you created on GitHub. It'll appear with a copy button and everything, and end in .git
```
git remote add origin remote_repository_URL_here
```

Verify the new remote URL
```
git remote -v
```

Set upstream and then push only the lfs files to remote
```
git lfs push origin master
```

Afterwards push normally to upload everything
```
git push --set-upstream origin master
```

You only need to write --set-upstream origin master the first time for normal `push`, after this just write push. For git lfs you always have to write it.

<a id="manage-multiple-github-or-gitlab-accounts"></a>
### Manage multiple GitHub or GitLab accounts

Because I want to update my personal code when I find better ways to program at work, I want to push and pull from my personal GitHub account aside from the work GitLab projects. **CAUTION: DON'T UPLOAD COMPANY SECRETS TO YOUR PERSONAL ACCOUNT**

To be able to do this, I followed these guides:<br>
https://blog.gitguardian.com/8-easy-steps-to-set-up-multiple-git-accounts/


https://medium.com/the-andela-way/a-practical-guide-to-managing-multiple-github-accounts-8e7970c8fd46


https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token


1. Generate an SSH key
First, create an SSH key for your personal account:
```
ssh-keygen -t rsa -b 4096 -C "your_personal_email@example.com" -f ~/.ssh/<personal_key> 
```
Then for your work account:
```
ssh-keygen -t rsa -b 4096 -C "your_work_email@company.com" -f ~/.ssh/<work_key> 
```

2. Add a passphrase

Then add a passphrase and press enter, it will ask for it twice. Press enter again.

To update the passphrase for your SSH keys:

```
ssh-keygen -p -f ~/.ssh/<personal_key>
```

You can check your newly created key with:

```
ls -la ~/.ssh
```

which should output <personal_key> and <personal_key>.pub.

Do the same steps for the <work_key>.

3. Tell ssh-agent

The website has an -K tag that works for macOSX and such but we don't need it.

```
eval "$(ssh-agent -s)" && \
ssh-add ~/.ssh/<personal_key> 
ssh-add ~/.ssh/<work_key> 
```

4. Edit your SSH config

```
nano ~/.ssh/config
-----------nano----------
# Work account - default
Host <some_host_name_work>
  HostName <HOST>:<PORT>
  User git
  IdentityFile ~/.ssh/<work_key> 

# Personal account
Host <personal_host_name>
  HostName github.com
  User git
  IdentityFile ~/.ssh/<personal_key>

CTRL+O
CTRL+X
-------------------------
```

5. Copy the SSH public key

```
cat ~/.ssh/<personal_key>.pub | pbcopy
```

Then paste on your respective website settings, such as the GitHub SSH settings page. Title it something you'll know it's your work computer.

Same for your <work_key>

6. Structure your workspace for different profiles

Now, for each key pair (aka profile or account), we will create a .conf file to make sure that your individual repositories have the user settings overridden accordingly.
Let’s suppose your home directory is like that:

```
/myhome/
    |__.gitconfig
    |__work/
    |__personal/
```

We are going to create two overriding .gitconfigs for each dir like this:

```
/myhome/
|__.gitconfig
|__work/
     |_.gitconfig.work
|__personal/
    |_.gitconfig.pers
```

Of course the folder and filenames can be whatever you prefer.

7. Set up your Git configs

In the personal git projects folder, make `.gitconfig.pers`

```
nano ~/personal/.gitconfig.pers
---------------nano-----------------
# ~/personal/.gitconfig.pers
 
[user]
email = your_personal_email@example.com
name = Your Name
 
[github] #or gitlab or whatever
user = "personal-username"
 
[core]
sshCommand = “ssh -i ~/.ssh/<personal_key>”

```


```

# ~/work/.gitconfig.work
 
[user]
email = your_work_email@company.com
name = Your Name
 
[github] #or gitlab or whatever
user = "work_username"

[core]
sshCommand = “ssh -i ~/.ssh/<work_key>”


```

And finally add this to the end of your original main `.gitconfig` file:

```
[includeIf “gitdir:~/personal/”] # include for all .git projects under personal/ 
path = ~/personal/.gitconfig.pers
 
[includeIf “gitdir:~/work/”]
path = ~/work/.gitconfig.work
```

Now finally to confirm if it worked, go to any work project you have and type the following:

```
cd ~/work/work-project
git config user.email
```

It should be your work e-mail.

Now go to a personal project:

```
cd ~/personal/personal-project
git config user.email
```

And it should output your personal e-mail.

8. **To clone new projects**, specially private or protected ones, use the username before the website:

```
git clone https://<username>@github.com/<organization>/<repo>.git
```

If you have a 2 Factor Authentication, the clone might fail on the first try, because you need to generate a Personal Access Token.

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

And then copy and paste that as the password when the terminal asks you for user and password.s

And done! When you push or pull from the personal account you might encounter some 2 factor authorizations at login, but otherwise it's ready to work on both personal and work projects.


<a id="install-docker-desktop-for-mac"></a>
## Install Docker Desktop for Mac

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/desktop/install/mac-install/

- Use Apple menu >About this Mac. 
- Confirm the Chip to determine the installation method

The desktop app needs to be running before being able to run commands on the terminal.

Test:`docker container --list`

<a id="install-python-versions-with-pyenv-and-virtual-environments-with-poetry"></a>
## Install Python versions with pyenv and virtual environments with poetry

<a id="useful-data-science-libraries"></a>
### Useful Data Science libraries

This is my generic fresh start install so I can work. Usually I'd install all of them in general, but recently I only install the necessary libraries under venv. There's more libraries with complicated installations in other repositories of mine, and you might not wanna run this particular piece of code without checking what I'm doing first. For example, you might have a specific version of Tensorflow that you want, or some of these you won't use. But I'll leave it here as reference.

<a id="basic-tasks"></a>
#### Basic tasks:

```
pip install numpy scipy jupyter statsmodels \
pandas pathlib tqdm retry openpyxl
```

<a id="plotting"></a>
#### Plotting:
```
pip install matplotlib adjustText plotly kaleido
```

<a id="basic-data-science-and-machine-learning"></a>
#### Basic data science and machine learning:
```
pip install sklearn sympy pyclustering
```

<a id="data-mining--text-mining--crawling--scraping-websites"></a>
#### Data mining / text mining / crawling / scraping websites:
```
pip install beautifulsoup4 requests selenium
```

<a id="natural-language-processing-nlp"></a>
#### Natural language processing (NLP):
```
pip install gensim nltk langdetect
```

For Japanese NLP tools see:
https://github.com/elisa-aleman/MeCab-python

For Chinese NLP tools see:
https://github.com/elisa-aleman/StanfordCoreNLP_Chinese

<a id="neural-network-and-machine-learning"></a>
#### Neural network and machine learning:
```
pip install tensorflow tflearn keras \
torch torchaudio torchvision \
optuna
```

<a id="xgboost"></a>
#### XGBoost

To Install with CPU:
```
pip install xgboost
```

<a id="lightgbm"></a>
#### LightGBM

Install with CPU:

```
pip install lightgbm
```

<a id="minepy--maximal-information-coefficient"></a>
#### MINEPY / Maximal Information Coefficient

For Minepy / Maximal Information Coefficient, we need the Visual Studio C++ Build Tools as a dependency, so install it first:<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

```
pip install minepy
```

<a id="computer-vision-opencv"></a>
#### Computer Vision (OpenCV)

**Note to self: re-write with poetry project use instead of venv**

with CPU and no extra options:

```
python -m pip install -U opencv-python opencv-contrib-python
```


<a id="shell-scripting-for-convenience"></a>
## Shell Scripting for convenience

When it comes down to it, specially when working with LaTeX or git, you find yourself making the same commands over and over again. That takes time and frustration, so I find that making scripts from time to time saves me a lot of time in the future.


<a id="basic-flag-setup-with-getopts"></a>
### Basic flag setup with getopts

Once in a while those scripts will need some input to be more useful in as many cases as possible instead of a one time thing.

Looking for how to do this I ran across [a simple StackOverflow question](https://stackoverflow.com/questions/14447406/bash-shell-script-check-for-a-flag-and-grab-its-value), which led me to the `getopts` package and its tutorial:

[Getopts manual](https://archive.ph/TRzn4)

This is a working example:

```
while getopts ":a:" opt; do
  case $opt in
    a)
      echo "-a was triggered, Parameter: $OPTARG" >&2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

```

<a id="argparse-bash-by-nhoffman"></a>
### Argparse-bash by nhoffman

Now sometimes you'll want to have fancy arguments with both a shortcut name (-) and a long name (--), for example `-a` and `--doall` both pointing to the same command. In that case I recommend using nhoffman's implementation of Python's `argparse` in bash:

[argparse.bash by nhoffman on GitHub](https://github.com/nhoffman/argparse-bash)

---

That is all for now. This is my initial setup for the lab environment under a proxy. If I have any projects that need further tinkering, that goes on another repository / tutorial.

