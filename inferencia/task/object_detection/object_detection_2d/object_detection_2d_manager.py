from .label.object_detection_2d_label_factory import ObjectDetection2DLabelFactory
from .model.object_detection_2d_model_factory import ObjectDetection2DModelFactory
from .visualization.object_detection_2d_visualizer_factory import ObjectDetection2DVisualizerFactory


class ObjectDetection2DManager():

    def get_model(model_name="YoloV4Middle",
                  model_path=None,
                  model_precision="FP32",
                  conf_thresh=0.2,
                  nms_thresh=0.4,
                  label_name="COCO"):
        return ObjectDetection2DModelFactory.create(model_name,
                                                    model_path,
                                                    model_precision,
                                                    conf_thresh,
                                                    nms_thresh,
                                                    label_name)

    def get_visualizer(visualizer_name="BBOXVisualizer"):
        return ObjectDetection2DVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return ObjectDetection2DLabelFactory.create(label_name)
