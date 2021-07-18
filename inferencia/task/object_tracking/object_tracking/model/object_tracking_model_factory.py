import json
# from .object_tracking
from .object_tracking_model_name import ObjectTrackingModelName


class ConfigDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        json.JSONDecoder.__init__(self, **kwargs)
        # Use the custom JSONArray
        self.parse_array = self.JSONArray
        # Use the python implemenation of the scanner
        self.scan_once = json.scanner.py_make_scanner(self)

    def JSONArray(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        return tuple(values), end


class ObjectTrackingModelFactory():
    def create(object_tracking_model_name: str,
               multi_tracker_config_path: str,
               object_detection_model_name: str,
               use_iou_matching: bool,
               use_feature_extractor: bool,
               use_kalman_filter: bool,
               feature_extractor_name: str,
               input_fps: int,
               target_fps: int):
        if object_tracking_model_name == ObjectTrackingModelName.fastmot.value:
            from .model.fastmot.fastmot_model import FastMOTModel
            with open(multi_tracker_config_path) as cfg_file:
                config = json.load(cfg_file, cls=ConfigDecoder)

            multi_tracker_config = config['mot']['multi_tracker']
            return FastMOTModel(object_detection_model_name,
                                multi_tracker_config,
                                use_iou_matching,
                                use_feature_extractor,
                                use_kalman_filter,
                                feature_extractor_name,
                                input_fps,
                                target_fps)
        else:
            msg = "{} is not implemented. Choose from {}.".format(object_tracking_model_name,
                                                                  ObjectTrackingModelName.values())
            raise NotImplementedError(msg)
