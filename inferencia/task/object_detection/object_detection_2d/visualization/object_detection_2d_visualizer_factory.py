from .object_detection_2d_visualizer_name import ObjectDetection2DVisualizerName


class ObjectDetection2DVisualizerFactory():
    def create(visualizer_name="BBOXVisualizer"):
        if visualizer_name == ObjectDetection2DVisualizerName.bbox_visualizer.value:
            from .visualization.bbox_visualizer import BBOXVisualizer
            return BBOXVisualizer()
        else:
            msg = "model_name is {}, but not implemented".format(
                visualizer_name)
            raise NotImplementedError(msg)
