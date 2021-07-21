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
                 target_fps: int,
                 max_hold_ret_num: int,
                 ):

        self.tracking_model = FastMOT(object_detection_model_name,
                                      multi_tracker_config,
                                      use_iou_matching,
                                      use_feature_extractor,
                                      use_kalman_filter,
                                      feature_extractor_name,
                                      input_fps,
                                      target_fps,
                                      max_hold_ret_num,
                                      )

    def forward(self,
                frame,
                frame_index):
        obj_trk_hist = self.tracking_model.step(frame,
                                                frame_index)
        return obj_trk_hist

    def inference(self,
                  frame,
                  frame_index):
        obj_trk_hist = self.forward(frame, frame_index)
        return obj_trk_hist
