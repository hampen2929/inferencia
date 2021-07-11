# Docker
## GPU
```
docker build -t inferencia_gpu ./
```

```
docker run \
    --runtime=nvidia \
    -it \
    --rm \
    --name inferencia_gpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/inferencia/:/workspace \
    --net=host \
    inferencia_gpu \
    /bin/bash
```
