# Docker
## GPU
```
docker build -t onnx_infer_gpu ./
```

```
docker run \
    --runtime=nvidia \
    -it \
    --rm \
    --name onnx_infer_gpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/onnx_infer/:/workspace \
    --net=host \
    onnx_infer_gpu \
    /bin/bash
```
