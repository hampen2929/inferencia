from .skelton_based_action_recognition_model_name import SkeltonBasedActionRecognitionModelName


class SkeltonBasedActionRecognitionModelFactory():
    def create(model_name,
               model_path,
               model_precision,
               conf_thresh,
               nms_thresh,
               label_name):
        if model_name == SkeltonBasedActionRecognitionModelName.lightgbm.value:
            from .model.lightgbm.lightgbm import LightGBM
            return LightGBM(model_path=model_path,
                            model_precision=model_precision,
                            conf_thresh=conf_thresh,
                            nms_thresh=nms_thresh,
                            label_name=label_name)
        else:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)
