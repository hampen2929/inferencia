# Docker
## CPU
```
docker build -t onnx_infer_cpu ./
```

```
docker run \
    -it \
    --rm \
    --name onnx_infer_cpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/onnx_infer/:/workspace \
    --net=host \
    onnx_infer_cpu \
    /bin/bash
```
