from dataclasses import dataclass
import numpy as np


@dataclass
class PoseEstimation2dResult:
    pose: np.ndarray
    pose_norm: np.ndarray
    outputs: np.ndarray
    heatmap: np.ndarray
    query_location: np.ndarray
    image_height: int
    image_width: int

    def set_draw_image(self, draw_image):
        self.draw_image = draw_image


# class PoseEstimation2dResult:
#     def __init__(self,
#                  pose: np.ndarray,
#                  pose_norm: np.ndarray,
#                  outputs: np.ndarray,
#                  image_height: int,
#                  image_width: int,
#                  image=None,
#                  ):
#         self.pose = pose
#         self.pose_norm = pose_norm
#         self.outputs = outputs
#         self.image_height = image_height


#     def set_draw_image(self, draw_image):
#         self.draw_image = draw_image
