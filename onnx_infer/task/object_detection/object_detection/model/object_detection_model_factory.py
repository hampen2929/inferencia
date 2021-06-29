from .model_names import ModelName


class ObjectDetectionModelFactory():
    def create(model_name="YoloV4",
               model_patpose_estimation_2dh=None,
               model_precision="FP32",
               conf_thresh=0.2,
               nms_thresh=0.4,
               label_name="COCO"):
        if model_name == ModelName.yolov4.value:
            from .yolo_v4.yolo_v4 import YoloV4
            return YoloV4(model_path=model_path,
                          model_precision=model_precision,
                          conf_thresh=conf_thresh,
                          nms_thresh=nms_thresh,
                          label_name=label_name)
        else:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)
