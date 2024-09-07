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