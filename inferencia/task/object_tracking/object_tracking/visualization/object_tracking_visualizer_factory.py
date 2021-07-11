from .object_tracking_visualizer_name import ObjectTrackingVisualizerName


class ObjectTrackingVisualizerFactory():
    def create(visualizer_name="TrackingVisualizer"):
        if visualizer_name == ObjectTrackingVisualizerName.tracking_visualizer.value:
            from .visualization.tracking_visualizer import TrackingVisualizer
            return TrackingVisualizer()
        else:
            msg = "model_name is {}, but not implemented".format(
                visualizer_name)
            raise NotImplementedError(msg)
