from .label.skelton_based_action_recognition_label_factory import SkeltonBasedActionRecognitionLabelFactory
from .model.skelton_based_action_recognition_model_factory import SkeltonBasedActionRecognitionModelFactory
from .visualization.skelton_based_action_recognition_visualizer_factory import SkeltonBasedActionRecognitionVisualizerFactory


class SkeltonBasedActionRecognitionManager():

    def get_model(model_name="LightGBM",
                  model_path=None,
                  model_precision="FP32",
                  conf_thresh=0.2,
                  nms_thresh=0.4,
                  label_name=""):
        return SkeltonBasedActionRecognitionModelFactory.create(model_name,
                                                                model_path,
                                                                model_precision,
                                                                conf_thresh,
                                                                nms_thresh,
                                                                label_name)

    def get_visualizer(visualizer_name="ActionVisualizer"):
        return SkeltonBasedActionRecognitionVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return SkeltonBasedActionRecognitionLabelFactory.create(label_name)
