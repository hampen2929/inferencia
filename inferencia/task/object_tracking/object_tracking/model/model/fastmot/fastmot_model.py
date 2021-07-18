import json
from .fastmot.fastmot import FastMOT


class FastMOTModel():
    def __init__(self,
                 object_detection_model_name: str,
                 multi_tracker_config: dict,
                 use_iou_matching: bool,
                 use_feature_extractor: bool,
                 use_kalman_filter: bool,
                 feature_extractor_name: str,
                 input_fps: int,
                 target_fps: int):

        self.tracking_model = FastMOT(object_detection_model_name,
                                      multi_tracker_config,
                                      use_iou_matching,
                                      use_feature_extractor,
                                      use_kalman_filter,
                                      feature_extractor_name,
                                      input_fps,
                                      target_fps
                                      )

    def forward(self, frame):
        self.tracking_model.step(frame)

    def inference(self, frame):
        self.forward(frame)
        tracking_results = self.tracking_model.tracker.tracks
        return tracking_results
