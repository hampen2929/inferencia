from .tmp_visualizer_name import TmpVisualizerName


class TmpVisualizerFactory():
    def create(visualizer_name="TmpVisualizer"):
        if visualizer_name == TmpVisualizerName.tmp_visualizer.value:
            from .visualization.tmp_visualizer import Visualizer
            return Visualizer()
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                visualizer_name,
                TmpVisualizerName.names
            )
            raise NotImplementedError(msg)
