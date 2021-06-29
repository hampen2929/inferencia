from .visualize_bbox import visualize_bbox


class ObjectDetectionVisualizer():
    def __init__(self,
                 bbox_color=(0, 75, 255),
                 text_color=(0, 0, 0)):
        self.bbox_color = bbox_color
        self.text_color = text_color

    def visualize(self, image, object_detection_results: list):
        for object_detection_result in object_detection_results:
            visualize_bbox(image,
                           object_detection_result.class_name,
                           object_detection_result.xmin,
                           object_detection_result.ymin,
                           object_detection_result.xmax,
                           object_detection_result.ymax,
                           object_detection_result.confidence,
                           bbox_color=self.bbox_color,
                           text_color=self.text_color)
        return image
