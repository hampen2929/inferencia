from .body_reid_visualizer_name import BodyReidVisualizerName


class BodyReidVisualizerFactory():
    def create(visualizer_name="BodyReidVisualizer"):
        if visualizer_name == BodyReidVisualizerName.body_reid_visualizer.value:
            from .visualization.reid_visualizer import ReidVisualizer
            return ReidVisualizer()
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                visualizer_name,
                BodyReidVisualizerName.names
            )
            raise NotImplementedError(msg)
