from .object_detection_2d_model_name import ObjectDetection2DModelName


class ObjectDetection2DModelFactory():
    def create(model_name,
               model_path,
               model_precision,
               conf_thresh,
               nms_thresh,
               label_name):
        if model_name == ObjectDetection2DModelName.yolo_v4_middle.value:
            from .model.yolo_v4.yolo_v4_middle import YoloV4Middle
            return YoloV4Middle(model_path=model_path,
                                model_precision=model_precision,
                                conf_thresh=conf_thresh,
                                nms_thresh=nms_thresh,
                                label_name=label_name)
        elif model_name == ObjectDetection2DModelName.tiny_yolo_v4.value:
            from .model.yolo_v4.tiny_yolo_v4 import TinyYoloV4
            return TinyYoloV4(model_path=model_path,
                              model_precision=model_precision,
                              conf_thresh=conf_thresh,
                              nms_thresh=nms_thresh,
                              label_name=label_name)
        else:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)
