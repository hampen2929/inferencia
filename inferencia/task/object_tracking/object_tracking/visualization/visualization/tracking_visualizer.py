from typing import Dict, List
import numpy as np

from ..object_tracking_visualizer import ObjectTrackingVisualizer
from .visualize_tracking import visualize_tracking


class TrackingVisualizer(ObjectTrackingVisualizer):

    def visualize(self,
                  image: np.ndarray,
                  object_tracking_history: Dict,
                  bbox_color=(0, 75, 255),
                  text_color=(0, 0, 0)) -> np.ndarray:
        for trakcking_id, object_detection_result in object_tracking_history.items():
            visualize_tracking(image,
                               object_detection_result.class_name,
                               object_detection_result.xmin,
                               object_detection_result.ymin,
                               object_detection_result.xmax,
                               object_detection_result.ymax,
                               object_detection_result.confidence,
                               trakcking_id,
                               bbox_color=bbox_color,
                               text_color=text_color)
        return image
