from dataclasses import dataclass
import numpy as np


@dataclass
class PoseEstimation2dResult:
    pose: np.ndarray
    image_height: int
    image_width: int
    pose_norm: np.ndarray = None
    outputs: np.ndarray = None
    heatmap: np.ndarray = None
    query_location: np.ndarray = None
    draw_image = None

    def set_draw_image(self, draw_image):
        self.draw_image = draw_image
