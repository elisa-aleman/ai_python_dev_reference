# Suggested Tools

This guide is under construction

- [ ] env cli tools
    - [ ] anyenv (mac and linux)
        - [ ] https://github.com/anyenv/anyenv
        - [ ] clone, add path for bash
        - [ ] `echo 'eval "$(anyenv init -)"' >> ~/.bash_profile`, remove the hyphen suggested.
        - [ ] A warning will appear if I don't have a manifest directory
        - [ ] `mkdir ~/.config`
        - [ ] `anyenv install --init`
        - [ ] Actually pyenv latest version does not work with it so don't use or recommend for pyenv
    - [ ] pyenv / pyenv-win
        - [ ] https://github.com/pyenv/pyenv
        - [ ] `pyenv install 3.11.7` since 3.12 is not yet compatible with many projects.
    - [ ] pipx
        - [ ] https://github.com/pypa/pipx
    - [ ] poetry / poetry plugin export
        - [ ] https://python-poetry.org/docs/#installing-with-pipx
        - [ ] `pipx install poetry && pipx inject poetry poetry-plugin-export`
    - [ ] toml-cli
        - [ ] `pipx install toml-cli`
    - [ ] CUDA 12 (CTranslate2 now supports too)
- [ ] Documents and planning
    - [ ] Mermaid
    - [ ] Markmap
- [ ] IDE / Editors / Terminals
    - [ ] Sublime Text
    - [ ] Windows Terminal + Cmder
        - [ ] https://medium.com/talpor/windows-terminal-cmder-%EF%B8%8F-573e6890d143 
        - https://windowsterminalthemes.dev/
        - Starting directory can be /mnt/d/...
        - Make sure it's the `C:\Windows\system32\wsl.exe -d Ubuntu` and not the `ubuntu.exe` profile.
        - `C:\Windows\system32\wsl.exe -d Ubuntu --exec bash -l` to start in bash instead of sh
- [ ] Accessibility Tools
    - [ ] not bionic reading
    - [ ] visual
- [ ] Others
 