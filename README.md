# inferencia
The implementation of onnx model for inference

# Docker
## CPU
```
cd docker/cpu/
docker build -t inferencia_cpu ./
```

```
docker run -it \
    --rm \
    --name inferencia_cpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/inferencia/:/workspace \
    --net=host \
    inferencia_cpu \
    /bin/bash
```

## GPU
```
cd docker/gpu/
docker build -t inferencia_gpu ./
```

```
docker run --gpus all -it \
    --rm \
    --name inferencia_gpu \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-uni \
    -v $HOME/workspace/inferencia/:/workspace \
    --net=host \
    inferencia_gpu \
    /bin/bash
```

```
docker exec -it inferencia_gpu /bin/bash
```

# setup
```
python setup.py develop
```

# Directory
## command
```
tree -d -L 2 -I '__pycache__'
```
## tree
```
inferencia
├── task
│   ├── action_recognition
│   │   ├── action_recognition
│   │   └── skelton_based_action_recognition
│   ├── face_recognition
│   │   ├── age_estimation
│   │   ├── emotion_recognition
│   │   ├── gaze_estimation
│   │   └── head_pose_estimation
│   ├── image_classification
│   │   └── image_classification
│   ├── image_generation
│   ├── image_super_resolution
│   │   └── single_image_super_resolution
│   ├── instance_segmentation
│   ├── object_detection
│   │   ├── object_detection_2d
│   │   └── object_detection_3d
│   ├── object_tracking
│   │   └── object_tracking
│   ├── person_reid
│   │   └── body_reid
│   ├── pose_estimation
│   │   ├── pose_estimation_2d
│   │   └── pose_estimation_3d
│   └── tmp
│       └── tmp
└── util
    ├── color
    ├── file
    ├── formatter
    ├── label
    ├── logger
    ├── pre_process
    └── reader
        └── reader
```
## abstract tree
```
inferencia
└── task
    └── main_task
        └── sub_task
            ├── sub_task_manager.py
            ├── label
            │   ├── sub_task_label_factory.py
            │   ├── sub_task_label_name.py
            │   └── label
            │       └── sub_task_label.py
            ├── model
            │   ├── sub_task_model_factory.py
            │   ├── sub_task_model_name.py
            │   ├── sub_task_model.py
            │   ├── sub_task_result.py
            │   └── model
            │       └── method
            │           └── method_model.py
            └── visualization
                ├── sub_task_visualizer_factory.py
                ├── sub_task_visualizer_name.py
                ├── sub_task_visualizer.py
                └── visualization
                    └── sub_task_visualizer.py
```

# Develop
Branch is associate to one model

- Model develop
  - Directory
    - task/main_task/model/model/method/


  - Rule
    - MethodModel class in method_model.py inherits SubTaskModel class in sub_task_model.py
    - Add method name to SubTaskModelName in sub_task_model_name.py
    - Add method to SubTaskModelFactory in sub_task_model_factory.py
    - Add method to SubTaskManager in sub_task_manager.py

  - Other
    - Add label and visualize if need.

- Test
  - Must
    - MethodModel input and output
    - SubTaskModelName
    - SubTaskModelFactory
    - SubTaskManager



