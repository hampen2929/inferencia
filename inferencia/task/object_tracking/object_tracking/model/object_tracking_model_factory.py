from .object_tracking
_model_name import ObjectTrackingModelName


class ObjectTrackingModelFactory():
    def create(model_name,
               model_path,
               model_precision,
               conf_thresh,
               nms_thresh,
               label_name):
        if model_name == ObjectTrackingModelName.deep_sort.value:
            from .model import DeepSORT
            return DeepSORT(model_path=model_path,
                            model_precision=model_precision,
                            conf_thresh=conf_thresh,
                            nms_thresh=nms_thresh,
                            label_name=label_name)
        else:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)
