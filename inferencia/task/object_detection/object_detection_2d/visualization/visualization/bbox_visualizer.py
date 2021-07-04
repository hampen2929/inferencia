import numpy as np

from ..object_detection_2d_visualizer import ObjectDetection2DVisualizer
from .visualize_bbox import visualize_bbox


class BBOXVisualizer(ObjectDetection2DVisualizer):

    def visualize(self,
                  image: np.ndarray,
                  object_detection_results: list,
                  bbox_color=(0, 75, 255),
                  text_color=(0, 0, 0)) -> np.ndarray:
        for object_detection_result in object_detection_results:
            visualize_bbox(image,
                           object_detection_result.class_name,
                           object_detection_result.xmin,
                           object_detection_result.ymin,
                           object_detection_result.xmax,
                           object_detection_result.ymax,
                           object_detection_result.confidence,
                           bbox_color=bbox_color,
                           text_color=text_color)
        return image
