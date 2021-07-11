from .label.object_tracking_label_factory import ObjectTrackingLabelFactory
from .model.object_tracking_model_factory import ObjectTrackingModelFactory
from .visualization.object_tracking_visualizer_factory import ObjectTrackingVisualizerFactory


class ObjectTrackingManager():

    def get_model(model_name="SORT",
                  conf_thresh=0.2,
                  label_name="COCO"):
        return ObjectTrackingModelFactory.create(model_name,
                                                 conf_thresh,
                                                 label_name)

    def get_visualizer(visualizer_name="TrackingVisualizer"):
        return ObjectTrackingVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return ObjectTrackingLabelFactory.create(label_name)
