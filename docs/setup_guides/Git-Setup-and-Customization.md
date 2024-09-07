# Git setup guide

TODO:
- [ ] Update "Manage multiple accounts" section with newer knowledge
- [ ] Check all syntax highlights and location comments
- [ ] Add style guide for location comments
- [ ] Wrap all links

First install git according to your OS. 

## Install Git

### Git on Windows

Use Git Bash as described in the [Windows setup document](./Windows-Setup.md#install-git-bash)


I followed this guide for using git on Windows:<br>
https://www.pluralsight.com/guides/using-git-and-github-on-windows

Download and Install from: <br>
https://git-scm.com/download/win


### Git on MacOSX

Mac already has git installed (that's how we installed homebrew too), but we can update it and manage it with homebrew.

```sh
# @ shell(mac_osx)

brew install git
```

### Git on Linux

Linux already has git installed, but we can update it and manage it with apt-get.

```sh
# @ shell(linux/wsl)

sudo apt-get update
sudo apt-get install git
```

## Initial setup

Then setup the configuration. Make an account at [GitHub](https://github.com) to get a username and email associated with Git. Type the settings on the terminal. My settings are like this:

```sh
# @ shell(linux/mac_osx/wsl)

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

```yaml
# @ ~/.gitconfig

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


Next, Git on Windows has a limitation on filenames compared to Linux or MacOSX. Here's an explanation found [here](https://stackoverflow.com/a/22575737)

> Git has a limit of 4096 characters for a filename, except on Windows when Git is compiled with msys. It uses an older version of the Windows API and there's a limit of 260 characters for a filename. 

So as far as I understand this, it's a limitation of msys and not of Git. You can read the details [here](https://github.com/msysgit/git/pull/110). 

You can circumvent this by using another Git client on Windows or set `core.longpaths` to true as explained in other answers.

`git config --system core.longpaths true` 

Git is build as a combination of scripts and compiled code. With the above change some of the scripts might fail. That's the reason for core.longpaths not to be enabled by default. 


The [windows documentation on longpaths](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd#enable-long-paths-in-windows-10-version-1607-and-later) has some more information: 

>>   Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from common Win32 file and directory functions. However, you must opt-in to the new behavior. 

A registry key allows you to enable or disable the new long path behavior. To enable long path behavior set the registry key at `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)`

Since we don't want to mess up the system settings, let's set it up with global config files for now:

```sh
# @ git_bash

git config --global core.longpaths true
```


## Favorite Custom commands

### Check your branches in git log history in a pretty line

This makes your history tree pretty and easy to understand inside of the terminal.
I found this in https://stackoverflow.com/a/35075021

```sh
# @ shell(linux/mac_osx/wsl)

git log --all --decorate --oneline --graph
```

Not everyone would be doing a git log all the time, but when you need it just remember: 
"A Dog" = git log --all --decorate --oneline --graph

Actually, let's set an alias:

```sh
# @ shell(linux/mac_osx/wsl)

git config --global alias.adog "log --all --decorate --oneline --graph"
```

This adds the following to the .gitconfig file:

```yaml
[alias]
        adog = log --all --decorate --oneline --graph
```

And you run it like:

```sh
# @ shell(linux/mac_osx/wsl)

git adog
```


### Push with tags: multi-line git alias

To add a multi-line alias, for example, push and then push the tags on one single command, use `'!git ... && git ...'` as a format:

Push with tags:

```sh
# @ shell(linux/mac_osx/wsl)

git config --global alias.pusht '!git push && git push --tags'
```


## Manage multiple GitHub or GitLab accounts

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


### WSL and Windows shared ssh keys for multiple git accounts

I generally will not do this and instead I will use GitBash for handling git, local windows editors to edit the files, and WSL just to execute code, making this unnecessary.

However if that's not what you want to do, here's how to mirror the ssh keys on WSL to use git over there as well.


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



## Guide for Git Large File System

This is for files larger than 50 MB to be able to be used in Git. Still, GitLFS has some limitations if you don't buy data packages to increase your usage limit. By default you get 1GB of storage and 1GB of bandwidth (how much you push or pull per month). For 5$USD, you can add a *data pack* that adds 50GB bandwith and 50GB Git LFS storage.

Now we need to install the git-lfs package to use it:

Linux:

```
sudo apt install git-lfs
```

MacOSX:

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
git push --set-upstream origin main
```

You only need to write --set-upstream origin master the first time for normal `push`, after this just write push. For git lfs you always have to write it.


