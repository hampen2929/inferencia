# Docker
## CPU
```
docker build -t inferencia_cpu ./
```

```
docker run \
    -it \
    --rm \
    --name inferencia_cpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/inferencia/:/workspace \
    --net=host \
    inferencia_cpu \
    /bin/bash
```
