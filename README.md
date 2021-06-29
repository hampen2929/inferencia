# onnx_infer
The implementation of onnx model for inference

# Docker
## CPU
```
cd docker/cpu/
docker build -t onnx_infer_cpu ./
```

## GPU
```
cd docker/gpu/
docker build -t onnx_infer_gpu ./
```

```
docker run --gpus all -it \
    --rm \
    --name onnx_infer_gpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/onnx_infer/:/workspace \
    --net=host \
    -p 8888:8888 \
    onnx_infer_gpu \
    /bin/bash
```

```
docker exec -it onnx_infer_gpu /bin/bash
```

# setup
```
python setup.py develop
```

# Directory
```
onnx_infer
├── task
│   ├── image_classification
│   │   ├── image_classification
│   │   │   ├── ResNet
│   │   │   └── EfficientNet
│   ├── object_detection
│   │   ├── face_detection
│   │   ├── person_detection
│   │   └── object_detection
│   │       ├── factory
│   │       ├── label
│   │       ├── model
│   │       │   ├── yolo_v4
│   │       │   └── center_net
│   ├── pose_estimation
│   │   ├── pose_estimation_2d
│   │       └── move_net
│   │   └── pose_estimation_3d
│   └── action_recognition
│       ├── cnn
│       ├── gcn
│       └── tree
├── util



PackageA
├── service
│   ├── A
│   ├── B
│   └── C
└── util
