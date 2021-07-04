# Docker
docker run --gpus all -it \
    --rm \
    --name inferencia_yolo \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    --net=host \
    inferencia_gpu \
    /bin/bash

# Pytorch2ONNX
## Convert
```
python demo_pytorch2onnx.py \
    yolov4.pth \
    data/dog.jpg \
    8 80 416 416
```

```
python demo_pytorch2onnx.py \
    yolov4.pth \
    data/dog.jpg \
    0 80 416 416
```

sudo chown ubuntu:ubuntu /local/python-3.7.9/ -R