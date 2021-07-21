from .label.object_tracking_label_factory import ObjectTrackingLabelFactory
from .model.object_tracking_model_factory import ObjectTrackingModelFactory
from .visualization.object_tracking_visualizer_factory import ObjectTrackingVisualizerFactory


class ObjectTrackingManager():

    def get_model(
        multi_tracker_config_path,
        input_fps: int,
        target_fps: int,
        object_tracking_model_name="FastMOT",
        object_detection_model_name: str = "TinyYoloV4",
        feature_extractor_name: str = "osnet_x0_25",
        use_iou_matching: bool = True,
        use_feature_extractor: bool = True,
        use_kalman_filter: bool = True,
        max_hold_ret_num: int = 100
    ):
        return ObjectTrackingModelFactory.create(object_tracking_model_name=object_tracking_model_name,
                                                 multi_tracker_config_path=multi_tracker_config_path,
                                                 object_detection_model_name=object_detection_model_name,
                                                 use_iou_matching=use_iou_matching,
                                                 use_feature_extractor=use_feature_extractor,
                                                 use_kalman_filter=use_kalman_filter,
                                                 feature_extractor_name=feature_extractor_name,
                                                 input_fps=input_fps,
                                                 target_fps=target_fps,
                                                 max_hold_ret_num=max_hold_ret_num,
                                                 )

    def get_visualizer(visualizer_name="TrackingVisualizer"):
        return ObjectTrackingVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return ObjectTrackingLabelFactory.create(label_name)
