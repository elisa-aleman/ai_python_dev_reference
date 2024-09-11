Makefile for docker stuff escape $ with $

```
devrun:
    docker run -itd —ipc=host -v $${PWD%/*}:/work —name container_name$(suffix)_name image_name bash
```

Makefile rm image if existing 

```
devrm:
    docker stop name || true
    docker rm name || true 
```