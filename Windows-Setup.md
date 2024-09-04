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

<a id="easy-gitlab-or-github-math-add-paired--signs-to-the-keybinds"></a>
#### Easy GitLab or GitHub math: Add paired $ signs to the keybinds

I found myself needing paired $dollar signs$ for math expressions in either LaTeX, GitLab with KeTeX or GitHub also with a different syntax but still some interpretation of TeX.

Searching for [how to do it on macros](https://forum.sublimetext.com/t/snippet-wrap-current-line-or-selection/53285), I found this post about keybindings which is a way better solution:

https://stackoverflow.com/questions/34115090/sublime-text-2-trying-to-escape-the-dollar-sign

Which, as long as we implement the double escaped dollar sign solution, we can use freely.

- Preferences > Key Bindings:
- Add this inside the brackets:

```
// Auto-pair dollar signs
{ "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$$0\\$"}, "context":
    [
        { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
        { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
        { "key": "following_text", "operator": "regex_contains", "operand": "^(?:\t| |\\)|]|\\}|>|$)", "match_all": true },
        { "key": "preceding_text", "operator": "not_regex_contains", "operand": "[\\$a-zA-Z0-9_]$", "match_all": true },
        { "key": "eol_selector", "operator": "not_equal", "operand": "string.quoted.double", "match_all": true }
    ]
},
{ "keys": ["$"], "command": "insert_snippet", "args": {"contents": "\\$${0:$SELECTION}\\$"}, "context":
    [
        { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
        { "key": "selection_empty", "operator": "equal", "operand": false, "match_all": true }
    ]
},
{ "keys": ["$"], "command": "move", "args": {"by": "characters", "forward": true}, "context":
    [
        { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
        { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
        { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
    ]
},
{ "keys": ["backspace"], "command": "run_macro_file", "args": {"file": "Packages/Default/Delete Left Right.sublime-macro"}, "context":
    [
        { "key": "setting.auto_match_enabled", "operator": "equal", "operand": true },
        { "key": "selection_empty", "operator": "equal", "operand": true, "match_all": true },
        { "key": "preceding_text", "operator": "regex_contains", "operand": "\\$$", "match_all": true },
        { "key": "following_text", "operator": "regex_contains", "operand": "^\\$", "match_all": true }
    ]
},
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

<a id="setup-git"></a>
## Setup Git

At first, we need to make sure of the path that `.gitconfig` will be saved by default so let's config the user and email through GitBash itself.

Run these commands and change in your username and email where appropriate.
```
git config --global user.name "Your_username"
git config --global user.email "your_email@company.com"
```

Next, Git on Windows has a limitation on filenames compared to Linux or MacOSX. Here's an explanation found [here](https://stackoverflow.com/a/22575737)

> Git has a limit of 4096 characters for a filename, except on Windows when Git is compiled with msys. It uses an older version of the Windows API and there's a limit of 260 characters for a filename. <br><br>
So as far as I understand this, it's a limitation of msys and not of Git. You can read the details here: https://github.com/msysgit/git/pull/110 <br><br>
You can circumvent this by using another Git client on Windows or set core.longpaths to true as explained in other answers.<br><br> 
`git config --system core.longpaths true` <br><br>
Git is build as a combination of scripts and compiled code. With the above change some of the scripts might fail. That's the reason for core.longpaths not to be enabled by default. <br><br> 
The windows documentation at https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd#enable-long-paths-in-windows-10-version-1607-and-later has some more information: <br><br>
>>   Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from common Win32 file and directory functions. However, you must opt-in to the new behavior. <br><br> A registry key allows you to enable or disable the new long path behavior. To enable long path behavior set the registry key at HKLM\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)

Since we don't want to mess up the system settings, let's set it up with global config files for now:

```
git config --global core.longpaths true
```

Now we can see that the `.gitconfig` file is under `"C:\Users\<user>\.gitconfig"`. I can now open that on SublimeText and paste my favorite settings for git:

```
# ~/.gitconfig

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
    longpaths = true
[pull]
    ff = only
[alias]
    adog = log --all --decorate --oneline --graph
```

That last one, `git adog` is very useful as I explain below:

<a id="check-your-branches-in-git-log-history-in-a-pretty-line"></a>
### Check your branches in git log history in a pretty line

This makes your history tree pretty and easy to understand inside of the terminal.
I found this in [https://stackoverflow.com/a/35075021](https://stackoverflow.com/a/35075021)

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


<a id="git-remote-origin-for-ssh"></a>
### Git remote origin for SSH

With SSH the structure would be this:
```
git clone ssh://git@<HOST>:<PORT>/username/your-project.git
```

**Create a new repository**

```
git clone http://<HOST>:<PORT>/username/your-project.git
cd your-project
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

**Existing folder**

```
cd existing_folder
git init
git remote add origin http://<HOST>:<PORT>/username/your-project.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

**Existing Git repository**

```
cd existing_repo
git remote rename origin old-origin
git remote add origin http://<HOST>:<PORT>/username/your-project.git
git push -u origin --all
git push -u origin --tags
```

To change from SSH to HTTP:
```
git remote set-url origin http://<HOST>:<PORT>/username/your-project.git
```

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
cat ~/.ssh/<personal_key>.pub | clip
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

And done! When you push or pull from the personal account you might encounter some 2 factor authorizations at login, but otherwise it's ready to work on both personal and work projects.

<a id="wsl-and-windows-shared-ssh-keys"></a>
#### WSL and Windows shared ssh keys

Do the whole thing on windows first, then follow these steps:

https://devblogs.microsoft.com/commandline/sharing-ssh-keys-between-windows-and-wsl-2/

1. Copy keys to WSL

```
cp -r /mnt/c/Users/<username>/.ssh ~/.ssh
```

2. Update permissions on the keys

```
chmod 600 ~/.ssh/id_rsa
```
Repeat for the other keys as well

3. Install keychain

```
sudo apt install keychain
```

4. Add keychain eval to .bash_profile for every key you have:

```
echo 'eval "$(keychain --eval --agents ssh id_rsa)"' >> ~/.bash_profile
echo 'eval "$(keychain --eval --agents ssh id_rsa_<personal_key>)"' >> ~/.bash_profile
echo 'eval "$(keychain --eval --agents ssh id_rsa_<work_key>)"' >> ~/.bash_profile
```

This will make it so that every time you start up the computer you have to type in the passwords for each of the keys, but they'll remain accessible after that.

5. Setup Git config to match

Copy .gitconfig from the Windows home to the WSL home folder.

Mirror the folder structure and sub-configuration files (e.g. .gitconfig.pers, .gitconfig.work).

Now any new folders created under WSL in these folders will have the same permissions.

However, if you want to access a git repository under the Windows environment through WSL, entering the paths to match will not be enough.

For example, even if you add: 

```
[includeIf “gitdir:/mnt/c/Users/<username>/personal/”] # include for all .git projects under personal/ 
path = /mnt/c/Users/<username>/personal/.gitconfig.pers
```
 Git will return an error like this:

```
fatal: detected dubious ownership in repository at '/mnt/c/Users/......'
To add an exception for this directory, call:

        git config --global --add safe.directory /mnt/c/Users/......
```

This happens because the path to the directory is different than expected, even if it points at the same directory.

https://stackoverflow.com/questions/73485958/how-to-correct-git-reporting-detected-dubious-ownership-in-repository-withou

This link explains that the newer versions of git are stricter with directory ownership.

This can be bypassed by setting this: **(However, only use this if you do not consider yourself at risk)**

```
git config --global safe.directory '*'
```

Now it is accessible from both ends!

If you mirrored the folders as well as added the windows folders, your configuration file should look like this:

```
[includeIf “gitdir:~/personal/”] # include for all .git projects under personal/ 
path = ~/personal/.gitconfig.pers
[includeIf “gitdir:/mnt/c/Users/<username>/personal/”] # include for all .git projects under personal/ 
path = /mnt/c/Users/<username>/personal/.gitconfig.pers
```


<a id="install-docker-desktop-for-windows"></a>
## Install Docker Desktop for Windows

Docker allows us to run server apps that share an internal environment separate from the OS.

Follow the following guide for docker.
https://docs.docker.com/desktop/install/windows-install/

Reboot after installing.

Running docker on WSL is also possible while having the Docker Desktop app open.
The desktop app needs to be running before being able to run commands on the shell (WSL recommended).

Test:`docker container --list`

<a id="install-python-versions-with-pyenv-win-and-virtual-environments-with-poetry"></a>
## Install Python versions with pyenv-win and virtual environments with poetry

<a id="useful-data-science-libraries"></a>
### Useful Data Science libraries

This is my generic fresh start install list so I can work. I only install the necessary libraries under poetry, so I don't recommend copy pasting all of it. There's more libraries with complicated installations in other repositories of mine, and you might not wanna run this particular piece of code without checking what I'm doing first. For example, you might have a specific version of Tensorflow that you want, or some of these you won't use. But I'll leave it here as reference.

<a id="basic-tasks"></a>
#### Basic tasks:

```
poetry add numpy scipy statsmodels \
pandas pathlib tqdm retry openpyxl
```

<a id="plotting"></a>
#### Plotting:
```
poetry add matplotlib adjustText plotly kaleido
```

<a id="basic-data-science-and-machine-learning"></a>
#### Basic data science and machine learning:
```
poetry add sklearn sympy pyclustering
```

<a id="data-mining--text-mining--crawling--scraping-websites"></a>
#### Data mining / text mining / crawling / scraping websites:
```
poetry add beautifulsoup4 requests selenium
```

<a id="natural-language-processing-nlp"></a>
#### Natural language processing (NLP):
```
poetry add gensim nltk langdetect
```

For Japanese NLP tools see:
https://github.com/elisa-aleman/MeCab-python

For Chinese NLP tools see:
https://github.com/elisa-aleman/StanfordCoreNLP_Chinese

<a id="neural-network-and-machine-learning"></a>
#### Neural network and machine learning:
```
poetry add tensorflow tflearn keras \
torch torchaudio torchvision \
optuna
```

<a id="xgboost"></a>
#### XGBoost

To Install with CPU:
```
poetry add xgboost
```

<a id="lightgbm"></a>
#### LightGBM

Install with CPU:

```
poetry add lightgbm
```

<a id="minepy--maximal-information-coefficient"></a>
#### MINEPY / Maximal Information Coefficient

For Minepy / Maximal Information Coefficient, we need the Visual Studio C++ Build Tools as a dependency, so install it first:<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

```
poetry add minepy
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

That is all for now. This is my initial setup for the development environment. If I have any projects that need further tinkering, that goes on another repository / tutorial.

