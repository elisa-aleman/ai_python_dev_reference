# Suggested Tools and Setup

This guide is under construction


- [ ] IDE / Editors
    - [ ] Sublime Text recheck settings

- [ ] env cli tools recheck all commands 
    - [ ] Add docker
- [ ] write about python AI
          CUDA 12.1 (CTranslate2 now supports too) repos install methods
          maybe write about diff between torch versions. 
 
- [ ] Documents and planning
    - [ ] Mermaid
    - [ ] Markmap
- [ ] Re check all commands
- [ ] syntax colors 
- [ ] Once guide is finished propagate relevant parts to OS specific guides with links to this main document

 

## IDE / Editors

### Sublime Text

- Install SublimeText4 for ease of use (this is my personal favorite, but it's not necessary)

#### Install SublimeText on Windows and MacOSX


https://www.sublimetext.com/download


#### Install SublimeText on linux


https://www.sublimetext.com/docs/linux_repositories.html

```
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/sublimehq-archive.gpg

sudo apt-get update
sudo apt-get install sublime-text
```

#### Setup SublimeText

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


### Sublime Merge

## Terminals

- [ ] Terminals
    - [ ] Windows Terminal + Cmder
        - [ ] https://medium.com/talpor/windows-terminal-cmder-%EF%B8%8F-573e6890d143 
        - https://windowsterminalthemes.dev/
        - Starting directory can be /mnt/d/...
        - Make sure it's the `C:\Windows\system32\wsl.exe -d Ubuntu` and not the `ubuntu.exe` profile.
        - `C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l` to start in bash instead of sh]
    - [ ] iTerm2


### Windows

### MacOSX

### Linux


 
## Environment and CLI tools

### anyenv

- [ ] anyenv (mac and linux)
        - [ ] https://github.com/anyenv/anyenv
        - [ ] clone, add path for bash
        - [ ] `echo 'eval "$(anyenv init -)"' >> ~/.bash_profile`, remove the hyphen suggested.
        - [ ] A warning will appear if I don't have a manifest directory
        - [ ] `mkdir ~/.config`
        - [ ] `anyenv install --init`
        - [ ] Actually pyenv latest version does not work with it so don't use or recommend for pyenv


### pyenv / pyenv-win

TODO: RE-check tutorial

Depending on your installation, you might already have a python, but it is better to avoid using it as it interacts with the system, so we install a local version with Pyenv. Pyenv also makes it so that pip and python are always matched for each other in the correct version.

This is specially useful if you need different versions for different projects (Maybe caused by incompatible updates).

- [ ] pyenv / pyenv-win
        - [ ] https://github.com/pyenv/pyenv
        - [ ] `pyenv install 3.11.7` since 3.12 is not yet compatible with many projects.


#### pyenv for Linux and MacOSX


https://github.com/pyenv/pyenv#installation

For Linux and MacOSX natively, and for Linux on WSL on Windows, we have to use the github distribution, clone it and install it.

Then we add the paths to `.bash_profile` for bash or to `.zprofile` for zsh.

```
cd ~
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
cd ~

source ~/.bash_profile
```

And then we can add it to our PATH so that every time we open `python` it's the pyenv one and not the system one:

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile

source ~/.bash_profile
```


#### pyenv-win for Windows

If for some reason you are not using WSL for development (which I don't recommend), pyenv-win, which is a fork of pyenv specifically for installing windows versions of python. The benefit of this is it lets you install several versions of python at the same time and use different ones in different projects, without path conflicts.

Let's follow the installation guide here:<br>
https://github.com/pyenv-win/pyenv-win#installation

```
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
```

The next step is to add the pyenv paths to the PATH environmental variable so the git bash reads them correctly.

There is two ways of doing this, but I chose option 1 because it's faster and can be copied to new machines.

**Option 1) Adding the paths to .bashrc so they're added each time Git Bash is opened**

```
nano ~/.bashrc

---- add in nano interface---
# PYENV paths
export PATH=$PATH:~/.pyenv/pyenv-win/bin:~/.pyenv/pyenv-win/shims
export PYENV=~/.pyenv/pyenv-win/
export PYENV_ROOT=~/.pyenv/pyenv-win/
export PYENV_HOME=~/.pyenv/pyenv-win/
export PYENV_PYTHON_EXE=$(dirname $(pyenv.bat which python))
export PATH=$PATH:$PYENV_PYTHON_EXE
# To update PYENV_PYTHON_EXE if pyenv changes versions close bash and open again

CTRL+O
CTRL+X
--------------------
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
```
%USERPROFILE%\.pyenv\pyenv-win\bin
%USERPROFILE%\.pyenv\pyenv-win\shims
```

Click OK until all the windows go away.


**Restart Git Bash**

```
echo $PYENV
echo $PATH
```

And the new environmental variables should be added.

Now in Git Bash check if it works

```
pyenv --version
```

Go to System Properties, search for Environmental Variables and click the Environmental Variables button that results from an old fashioned settings window.

From the User Path variable: delete `%USERPROFILE%\AppData\Local\Microsoft\WindowsApps`, which is conflicting with the `python` command for some reason.

Click OK until all the windows go away.

**Restart Git Bash**

Now if we just run `python` in Git Bash, it will hang instead of opening the interpreter. In order to run Python in Git Bash the same as we did in Unix based systems, we have to go back to the `.bashrc` and add a path to the a new alias.

```
nano ~/.bashrc
----nano interface---
# To run Python in Git Bash like in Unix
alias python='winpty python.exe'
CTRL+O
CTRL+X
source ~/.bashrc
--------------------
```

\* Solution gotten from:<br>
https://qiita.com/birdwatcher/items/acbc79005d24616de5b6

Now typing `python` in Git Bash should open the python interface without hanging.

To update pyenv-win, if installed via Git go to `%USERPROFILE%\.pyenv\pyenv-win` (which is your installed path) and run `git pull`.

#### Install Python with pyenv / pyenv-win

Now let's install and set the version we will use (`pyenv install -l` to check available versions).

```
pyenv install 3.10.7
pyenv global 3.10.7
```

We can confirm we are using the correct one:

```
pyenv versions
which python
python --version
which pip
pip --version
```


All these should now return pyenv versions.

Let's also upgrade pip:
```
pip install --upgrade pip
```


### pipx


- [ ] pipx
        - [ ] https://github.com/pypa/pipx


### Poetry for python project / dependency management

[Poetry](https://python-poetry.org/) 

TODO: rewrite section with pipx and docker added explanation

<!-- 
Poetry is a tool to manage python project dependencies and environments in a version controlled (e.g. git) and group accessible syntax. It allows to use a virtual environment to locally install all dependencies, remove or update them as needed while having access to previous instances of the environment at a given time via the commit history


- [ ] poetry / poetry plugin export
        - [ ] https://python-poetry.org/docs/#installing-with-pipx
        - [ ] `pipx install poetry && pipx inject poetry poetry-plugin-export`



Usage guide: https://python-poetry.org/

Making a new project can be as easy as:

```
poetry new project-name-here
cd project-name-here
```

Then, instead of using `pip install` or `pip uninstall` we use `poetry add`

```
poetry add pathlib
```

This updates the dependency control files `poetry.toml`, `poetry.lock`, and `pyproject.toml`, which can be committed to version control.

And finally, when cloning a repository, you can use `poetry install` to easily install all the dependencies controlled by poetry in one command.
 -->

### toml-cli


- [ ] toml-cli
        - [ ] `pipx install toml-cli`



## Documents and Planning

### Mermaid

### Markmap

### GitHub/GitLab flavored markdown math

#### GitHub Markdown math expressions for README.md, etc.

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


#### GitLab Markdown math expressions for README.md, etc.

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

### LaTeX for documentation / reports

I use LaTeX for writing detailed reports, curriculum vitae documents, resumes, academic publication history, or cover letters. The files are way lighter than whatever mess you can make with other word editors, and it gives me precise control over how the document will look in the end.

For every installation method described below, a few other useful packages aside from just LaTeX along with it, including `latexdiff` which I used a lot to compare revisions on academic papers, or `bibtex` for bibliography parsing, among others.

#### Installation

##### LaTeX on Windows

https://www.latex-project.org/

Because I'm used to Unix systems already, the most usable packaging that offers that similarity across systems seems to be TeXLive.

https://tug.org/texlive/windows.html

Download the installer and run:

https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe

Make sure default paper is A4.

Now after installation you might get some install failures for a few packages, and will get a command with `tlmgr` and some options to run. That command can only be run in the CMD prompt and not in Git Bash.

This should also include `latexdiff`.

I recommend running latex in git bash for version control.

If you need to configure Tex Live using `tlmgr`, it's necessary to run the commands in the CMD prompt. However, `pdflatex` as well as `bibtex` run normally in git bash, so feel free to use the same git bash for both compiling and version control.

##### LaTeX on MacOSX

Download the `.pkg` file and install it manually from the [mactex download website](https://www.tug.org/mactex/downloading.html)

It is installed under `/Library/TeX/texbin`, but if you have any terminal sessions open it will probably not load, so you should open a new session, and then confirm that it also installed `latexdiff` with:

```
which latexdiff
```

##### LaTeX on Linux

The installation on Linux is straight-forward:

The specific package for LaTeX on linux has different versions depending on how many of the accompanying packages you wish to install. [You can read about the differences here](https://tex.stackexchange.com/questions/245982/differences-between-texlive-packages-in-linux). I prefer `texlive-latex-extra`, one step below the largest installation of them all.

```
sudo apt install texlive-latex-extra
sudo apt-get install latexdiff
```


#### Use my LaTeX helper shell scripts for faster compilation

https://github.com/elisa-aleman/latex_helpers

I made these shell scripts to help in compiling faster when using bibliographies and to delete cumbersome files when not necessary every time I compile. Since they are .sh scripts, they run normally with git bash or the enhanced cmder on windows and run natively on shells in Linux and MacOSX.

Personally, I find it tiring to try to compile a LaTeX document, only to have to run the bibliography, and then compile the document twice again so all the references are well put where they need to be, rather tiring. Also, I find that the output files are cluttering my space and I only need to see them when I run into certain errors.

Also, for academic papers, I used `latexdiff` commands quite a lot, and while customizable, I noticed I needed a certain configuration for most journals and that was it.

So I made [LaTeX helpers](https://github.com/elisa-aleman/latex_helpers), a couple of bash scripts that make that process faster.

So instead of typing

```
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
open paper.tex
rm paper.log paper.out paper.aux paper.... and so on
```

Every. Single. Time. 

I just need to type:
```
./latexcompile.sh paper.tex --view --clean
```

and if I needed to make a latexdiff I just:

```
./my_latexdiff.sh paper_V1-1.tex paper.tex --newversion="2" --compile --view --clean
```

And there it is, a latexdiff PDF right on my screen.

I would also commonly have several documents of different languages, or save my latexdiff command in another script, called `cur_compile_all.sh` or `cur_latexdiff.sh` so I didn't have to remember version numbers and stuff when working across several weeks or months.

Usually with code such as:

```
cd en
./latexcompile.sh paper.tex --view --clean --xelatex
cd ../es
./latexcompile.sh paper.tex --view --clean --xelatex
cd ../jp
./latexcompile.sh paper.tex --view --clean --xelatex
```

And so on, to save time.


#### Make LaTeX easier in Sublime Text:

- Install Package Control.
- Install LaTeXTools plugin.

https://tex.stackexchange.com/a/85487

If you have the LaTeXTools plugin, it already does that except that it is mapped on Shift+Enter instead of Enter.

<a id="xelatex-in-japanese"></a>
#### XeLaTeX in Japanese

For Japanese UTF-8 text in XeLaTeX:

``` 
\usepackage{xeCJK}
```

Set the fonts: these are the default, but they have no bold
```
\setCJKmainfont{IPAMincho} % No bold, serif
\setCJKsansfont{IPAGothic} % No bold, sans-serif
\setCJKmonofont{IPAGothic} % No bold, sans-serif
```

Installing fonts, for example, Aozora mincho has guaranteed bold

https://web.archive.org/web/20200321102301/http://blueskis.wktk.so/AozoraMincho/download.html 

Make sure to install for all users:

https://stackoverflow.com/questions/55264642/how-to-force-win10-to-install-fonts-in-c-windows-fonts

Set the installed font:

```
\setCJKmainfont[BoldFont=AozoraMincho-bold,AutoFakeSlant=0.15]{Aozora Mincho}
```

Japanse document style:

```
\usepackage[english,japanese]{babel} % For Japanese date format
\usepackage{indentfirst} % For Japanese style indentation
\setlength\parindent{11pt}
```

Japanese babel messes itemize up inside tables, so:

```
\usepackage{enumitem}
\newlist{jpcompactitemize}{itemize}{1} % defined new list
\setlist[jpcompactitemize]{topsep=0em, itemsep=-0.5em, label=\textbullet} % new list setup
```


#### Display code sections in LaTeX

```
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[lighttt]{lmodern}
\usepackage{listings} % to display code
\usepackage{lstautogobble} % to indent inside latex without affecting the code, keeping the indent the code has inside
\usepackage{anyfontsize} % for code font size
\usepackage[os=win]{menukeys} % to display keystrokes

% For the color behind the code sections:
\usepackage{xcolor} %custom colours
\definecolor{light-gray}{gray}{0.95} %the shade of grey that stack exchange uses
\definecolor{editorGreen}{rgb}{0, 0.5, 0} % #007C00 -> rgb(0, 124, 0)

% Make a more defined languages for nice colors
\include{lststyle-css.sty}
\include{lststyle-html5.sty}

% Set up the code display lst options
\lstset{
    % for the code font and size:
    % basicstyle=\ttfamily\small,
    basicstyle=\ttfamily\fontsize{10}{12}\selectfont,
    % to avoid spaces showing as brackets in strings
    showstringspaces=false,
    % for straight quotes in code
    upquote=true, 
    % for the middle tildes in the code
    literate={~}{{\fontfamily{ptm}\selectfont \textasciitilde}}1,
    % for the line break in long texts
    breaklines=true,
    postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space}, 
    % for the keyword colors in the code
    keywordstyle=\color{blue}\bfseries\ttfamily,
    stringstyle=\color{purple},
    commentstyle=\color{darkgray}\ttfamily,
    keywordstyle={[2]{\color{editorGreen}\bfseries\ttfamily}},
    autogobble=true % to ignore latex indents but keep code indent
}

% unnecessary in XeLaTeX
% % For this specific document with lots of degree signs inside listings
% \lstset{
%     literate={°}{\textdegree}1
% }

% for straight double quotes in code
\usepackage[T1]{fontenc}

% frame set up
\usepackage[framemethod=TikZ]{mdframed} %nice frames
\mdfsetup{
    backgroundcolor=light-gray,
    roundcorner=7pt,
    leftmargin=1,
    rightmargin=1,
    innerleftmargin=1em,
    innertopmargin=0.5em,
    innerbottommargin=0,
    outerlinewidth=1,
    linecolor=light-gray,
    } 

% Make it affect all lstlistings
\BeforeBeginEnvironment{lstlisting}{\begin{mdframed}\vskip-.5\baselineskip}
\AfterEndEnvironment{lstlisting}{\end{mdframed}}

% Make colored box around inline code
\usepackage{realboxes}
\usepackage{xpatch}

\makeatletter
\xpretocmd\lstinline{\Colorbox{light-gray}\bgroup\appto\lst@DeInit{\egroup}}{}{}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```


### Install Pandoc to convert/export markdown, HTML, LaTeX, Word

I discovered this tool recently when I was asked to share a PDF of my private GitLab MarkDown notes. Of course I wouldn't share the whole repository so that it can be displayed in GitLab for them, so I searched for an alternative. 

It can be installed in Windows, macOS, Linux, ChromeOS, BSD, Docker, ... it's really portable

Pandoc Install:  
https://pandoc.org/installing.html


Pandoc Manual:  
https://pandoc.org/MANUAL.html


Export to PDF syntax
```
pandoc test1.md -s -o test1.pdf
```

Note that it uses LaTeX to convert to PDF, so UTF-8 languages (japanese, etc.) might return errors.

```
pandoc test1.md -s -o test1.pdf --pdf-engine=xelatex
```

But it doesn't load the Font for Japanese... Also, the default margins are way too wide.

So, in the original markdown file preamble we need to add [Variables for LaTeX](https://pandoc.org/MANUAL.html#variables-for-latex):

```
---
title: "Title"
author: "Name"
date: YYYY-MM-DD
<!-- add the following -->
geometry: margin=1.5cm
output: pdf_document
<!-- CJKmainfont: IPAMincho #default font but no bold -->
<!-- install this one for bold japanese: https://web.archive.org/web/20200321102301/http://blueskis.wktk.so/AozoraMincho/download.html -->
<!-- https://stackoverflow.com/questions/55264642/how-to-force-win10-to-install-fonts-in-c-windows-fonts (install for all users) -->
CJKmainfont: Aozora Mincho
CJKoptions:
- BoldFont=AozoraMincho-bold
- AutoFakeSlant=0.15
---
```

And voilà, the markdown is now a PDF.

I'm still unsure if it will process the GitHub or GitLab math environments, since the syntax is different.

Upon confirmation with the [User's Guide: Math](https://pandoc.org/MANUAL.html#math) section, it uses the GitHub math syntax.

Inline: `$x=3$`  
Renders as:  
$x=3$


Block:  `$$x=3$$`  
Renders as:  
$$x=3$$


## Web development

### Python FastAPI

Under construction

### Ruby, Bundler and Jekyll for static websites

Sometimes it might be necessary to present results in HTML websites. I also make websites on my free time, and lots of researchers have their projects on a github pages website. For this, I like to use Jekyll in combination with github pages.

For this, I like to use Jekyll, which needs Ruby and Bundler. 

#### Install Ruby on Windows

First lets install Ruby, on which Jekyll and Bundler will be later installed.

- Reference:
    - [Jekyll installation guide for windows](https://jekyllrb.com/docs/installation/windows/)

Install Ruby+Devkit on Windows with [RubyInstaller](https://rubyinstaller.org/)

- Don't forget to select the add Ruby related items to PATH option
- Run the ridk install at the last stage of the installation wizard.
- You'll be prompted to install MSYS2, I selected option 3.

Open a new command prompt window from the start menu, so that changes to the PATH environment variable becomes effective. 
`echo %PATH%` on cmd and `echo $PATH` on Git Bash should both have Ruby related paths now.


You can do this on the Git Bash too, but the output will be delayed, so run on cmd.

#### Install Ruby on MacOSX

- Reference:
    - [Jekyll installation guide for mac](https://jekyllrb.com/docs/installation/macos/)

First lets install Ruby, on which Jekyll and Bundler will be later installed.

```
brew install ruby
```

Then, we have to add to the $PATH so that ruby gems are found:

```
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="~/.local/share/gem/ruby/X.X.X/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

Where `X.X.X ` is the version you have installed.

In my case it was:

```
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="~/.local/share/gem/ruby/3.0.0/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

Check that we're indeed using the Homebrew version of ruby:

```
which ruby
```
Which should output this:
```
/usr/local/opt/ruby/bin/ruby
```


#### Install Ruby on Linux

- Reference:
    - [Jekyll installation guide for Ubuntu](https://jekyllrb.com/docs/installation/ubuntu/)

First lets install Ruby, on which Jekyll and Bundler will be later installed.

```
sudo apt-get install ruby-full build-essential zlib1g-dev
```

Then, we have to add to the $PATH so that ruby gems are found:

```
echo '# Install Ruby Gems to ~/gems' >> ~/.bash_profile
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bash_profile
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

#### Install Jekyll and Bundler

Install Jekyll and Bundler:

```
gem install bundler jekyll jekyll-sitemap
```

For MacOSX, it is necessary to install at the user level, if I recall correctly...

```
gem install --user-install bundler jekyll jekyll-sitemap
```

Check if installed properly `jekyll -v` on cmd or git bash.

Now it's installed! I'll be using this for local websites, but we're going to follow a tutorial on how to make Jekyll Github Pages, so even though we're not using GitHub, give this a read.

https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll

Make a new repository for your website on local git.

Now, once you have your webiste repository and you're ready to test the jekyll serve, do the following:

```
cd (your_repository_here)
bundle init
bundle add jekyll
bundle add jekyll-sitemap
bundle add webrick
```

And then all that's left to do is to serve the website with jekyll!
Also for the sitemaps make sure to check this tutorial:

https://github.com/jekyll/jekyll-sitemap

And add this to your `_config.yml`
```
url: "https://example.com" # the base hostname & protocol for your site
plugins:
  - jekyll-sitemap
```

```
bundle exec jekyll serve
```

If you get an error like:
```
Could not find webrick-1.7.0 in any of the sources
Run `bundle install` to install missing gems.
```

Do as it says and just run:
```
bundle install
```

Now you can work on the website and look at how it changes on screen.

By the way, if you are hosting on GitHub Pages and have a custom domain, you need to add these to the DNS

```
Type    Name    Points to               TTL
a       @       185.199.108.153         600 seconds
a       @       185.199.109.153         600 seconds
a       @       185.199.110.153         600 seconds
a       @       185.199.111.153         600 seconds
cname   www     your-username.github.io 600 seconds   
```

## Python suggested libraries

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

To Install with CUDA GPU integration:
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j8
cd ../python-package
python setup.py install
```

<a id="lightgbm"></a>
#### LightGBM

To Install with CPU:

```
pip install lightgbm
```

Install dependencies:
```
apt-get install libboost-all-dev
apt install ocl-icd-libopencl1
apt install opencl-headers
apt install clinfo
apt install ocl-icd-opencl-dev
```
Install with CUDA GPU integration:

```
pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
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


<a id="install-opencv-with-ffmpeg"></a>
##### Install OpenCV with ffmpeg

Install dependencies:

```
apt-get update
apt-get upgrade
apt-get install build-essential
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev
apt-get install libgtk-3-dev
apt-get install libatlas-base-dev gfortran pylint
apt-get install python2.7-dev python3.5-dev python3.6-dev
apt-get install unzip
```

Now the ffmpeg dependency:
```
add-apt-repository ppa:jonathonf/ffmpeg-3
apt update
apt install ffmpeg libav-tools x264 x265
```

Check the version:
```
ffmpeg
```

Download and build opencv

```
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
wget https://github.com/opencv/opencv_contrib/archive/3.4.0.zip -O opencv_contrib-3.4.0.zip

unzip opencv-3.4.0.zip
unzip opencv_contrib-3.4.0.zip

cd  opencv-3.4.0
mkdir build_3.5
mkdir build
cd build
```

Make, but remember to replace Python versions:

```
which python
```

```
cmake -DCMAKE_BUILD_TYPE=Release \
    -D WITH_FFMPEG=ON \
    -D PYTHON3_EXECUTABLE=<path to your python> \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
    -D OPENCV_ENABLE_NONFREE=True ..

make -j8        #(where -j8 is for 8 cores in the server cpu)
make install
ldconfig
```




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



## Accessibility Tools

### Accessible Color Palettes with Paletton

When designing new things it's important to keep in mind color theory, as well as accessibility for the visually impaired and color blind people, etc. But that's so much time one could spend doing so much else, so here's a tool that can help with that and also visualizing how other people with different ranges of color vision would perceive it. It's called Paletton.

https://paletton.com


### Reading tools for Neurodivergent people

There was a new tool developed called "Bionic Reading", which bolds the beginnings of words so that our eyes glide over them more easily, basically making a tool for speed reading without having to train specifically for that. Lots of neurodivergent people such as myself (I have ADHD and am autistic), have a hard time following long texts or focusing when there is too much information at the same time (say, with very small line spacing). This new tool has been praised by the ND (neurodivergent) community, since making it available for businesses or companies to use would mean more accessibility in everyday services...  or at least it was until they decided to charge an OUTRAGEOUS amount of money to implement it, making it obviously not attractive for companies to implement and therefore ruining it for everyone.

That is why someone decided to make "Not Bionic Reading" which is, legally speaking, not the same thing as Bionic Reading and therefore can be made available for everyone as Open Source.

Here's the usable link:
https://not-br.neocities.org/

Have fun reading!


### Reading white PDFs


#### Firefox

https://pncnmnp.github.io/blogs/firefox-dark-mode.html

> After hunting on the web for about 30 minutes, I found this thread on Bugzilla. It turns out starting with Firefox 60, extensions are no longer allowed to interact with the native pdf viewer. Determined, I decided to locally modify the CSS rendered by Firefox's PDF viewer. The steps for the same are:
>
> - Open Firefox and press Alt to show the top menu, then click on Help → Troubleshooting Information
> - Click the Open Directory button beside the Profile Directory entry
> - Create a folder named chrome in the directory that opens
> - In the chrome folder, create a CSS file with the name userContent.css
> - Open the userContent.css file and insert -

```
#viewerContainer > #viewer > .page > .canvasWrapper > canvas {
    filter: grayscale(100%);
    filter: invert(100%);
}
```

> - On Firefox's URL bar, type about:config.
> - Search for toolkit.legacyUserProfileCustomizations.stylesheets and set it to true.
> - Restart Firefox and fire up a PDF file to see the change!


#### Microsoft Edge

If you ever need to read a PDF on the Microsoft Edge browser, you can create a snippet to execute on the Dev Console, as per this post.

https://www.reddit.com/r/edge/comments/nhnflv/comment/hgejdwz/?utm_source=share&utm_medium=web2x&context=3

> So, I've wanted this feature pretty badly as well and I've found a workaround which doesn't involve inverting the whole OS but still takes some extra steps:
>
> 1. Open the PDF you want to read
> 2. Right click > Inspect Element
> 3. Select Console tab
> 4. Paste the code given below
> 5. Hit enter
> 6. Profit!


```
let backgroundColor = PDFViewer.EDGE_PDFVIEWER_BACKGROUND_COLOR_LIGHT;
viewer.plugin_.setAttribute('background-color', backgroundColor);
viewer.pluginController_.postMessage({
    type: 'backgroundColorChanged',
    backgroundColor
});
document.getElementById('document-container').style.filter = 'invert()';
document.getElementById('layout-container').style.filter = 'invert()';
```

> You can utilize the snippets feature in DevTools to save the above code. To do that, do:
> 
> 1. Hit F12 or Ctrl + Shift + I to open DevTools
> 2. Once the DevTools is open, press Ctrl + Shift + P and type "new snippet" and choose the first option
> 3. Paste the above code
> 4. Right click "Script snippet #2" > Rename > "dark mode pdf"
> 5. Hit enter to rename
> 6. Close DevTools
> 
> If you did the above to save that script, the next time, you can perform the following steps to activate dark mode:
> 
> 1. Open PDF
> 2. Right click > Inspect Element
> 3. Press Ctrl + P
> 4. Type exclamation "!"
> 5. Hit enter (or select the snippet if you have multiple and press enter)


#### Google Chrome

I found a solution in this post:

https://superuser.com/a/1527417


> The following snippet adds a div overlay to any browser tab currently displaying a PDF document.
> 1. Open up your browser's Dev tools then browser console.
> 2. Paste this JavaScript code in your browser console:

```
const overlay = document.createElement("div");

const css = `
    position: fixed;
    pointer-events: none;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: white;
    mix-blend-mode: difference;
    z-index: 1;
`
overlay.setAttribute("style", css);

document.body.appendChild(overlay);
```

> 3. Hit Enter
>
> Special thanks: https://www.reddit.com/r/chrome/comments/e3txhi/comment/fem1cto

 