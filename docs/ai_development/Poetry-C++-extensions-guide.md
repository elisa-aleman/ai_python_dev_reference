# Poetry python C++ and CUDA extensions guide

This guide is under development

Because poetry is still under development, some of this information is hidden in issues or discussions, but I've successfully built projects with dependencies that rely on source built extensions, or that the project itself relies on them. I will document here some examples as a sort of guide for future me, as well as other people to configure their projects.

TODO:
- [ ] write about pytorch CUDA wheels and the mess they were before torch 2.1 and CUDA 12.1
- [ ] Write about making the pyproject.toml build a repository wheel and sdist tar.gz file correctly with all the extensions there
- [ ] Write about exception cases, or non-PEP-517 compliant cases and what I did to solve it.
- [ ] Write about sources that are not pypi and also direct wheel usage
- [ ] write about the importance of WHL hash being the same for lock files generated within a docker container
- [ ] CUDA 12.1 (CTranslate2 now supports too) repos install methods
- [ ] Write about not updating to CUDA 12 too fast because of specific repositories not supporting it yet.
- [ ] Dockerfile examples
- [ ] 
- [ ] 
- [ ] 
- [ ] 

```
cd (project with c depends)
pip wheel --no-deps -w dist .
cd (my project)
poetry add /path/to/project/with/c/deps/dist/*.whl

```  


How to build whl for poetry project using local deps and c deps to add on another project


```
make docker container
check included packages found in setup.py
install dependencies apt
poetry init with options --no-interaction 
poetry version x.x.x  # match original version number
poetry add all requirements
use specific whl url sources instead of poetry source add
if errors thrown, use earlier versions
add custom built whl libraries local deps by filepath
make sure no symlinks on paths 
 MYWHL=$(readlink -f path/to/whl)
  poetry add $MYWHL

use toml-cli (pipx install toml-cli) to add to toml custom build dependencies 

!!always include setuptools in case of using this build.py method

toml set --toml-path pyproject.toml build-system.requires "[\"poetry-core\",\"numpy>=1.21\",\"cython\",\"setuptools\",\"packaging\",\"torch @ ${TORCH_WHL}\"]" --to-array

if its a local file whl make sure to do it in file URL  @ file:// format like so:

export TENSORRT_WHL="/root/workspace/TensorRT/python/...."

toml set --toml-path pyproject.toml build-system.requires "[\"poetry-core\",\"numpy>=1.21\",\"cython\",\"setuptools\",\"packaging\",\"torch @ ${TORCH_WHL}\",\"tensorrt @ file://${TENSORRT_WHL}\"]" --to-array

add the use of build.py to toml
toml add_section --toml-path pyproject.toml tool.poetry.build
toml set --toml-path pyproject.toml tool.poetry.build.script build.py
toml set --toml-path pyproject.toml tool.poetry.build.generate-setup-file true --to-bool

prepare a build.py that replaces setup.py extension functions:

----
from distutios.command.build_ext import build_ext
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
         name=,
....
)
]

def build(setup_kwargs):
    setup_kwargs.update(
         {
"ext_modules": ext_modules,
"cmdclass":{
       "build_ext": BuildExtension
    }
}
)
---

poetry build

which leaves a whl and a tar in /dist
whl might only include python and not c so we have to make sure the includes and paths build the C files IN the package root and not outside it

cd my project
poetry add /other/project/dist/tar-we-just-built

although currently mine throws errors when importing :(


```

<!--

```
docker run -itd --ipc=host -v ${PWD%/*}:/work --shm-size=4gb --gpus ‘“device=1”’ --name my_container python:3.11.7-bookworm bash

docker exec -it -w /work/project my_container bash

poetry new project-name
cd project-name
nano pyproject.toml
poetry install

exit

touch test.txt
docker exec -it -w /work/project my_container bash
ls -l

Check UID and GID

chown -R UID:GID ./*
```

-->