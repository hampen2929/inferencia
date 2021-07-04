from .skelton_based_action_recognition_visualizer_name import SkeltonBasedActionRecognitionVisualizerName


class SkeltonBasedActionRecognitionVisualizerFactory():
    def create(visualizer_name="BBOXVisualizer"):
        if visualizer_name == SkeltonBasedActionRecognitionVisualizerName.bbox_visualizer.value:
            from .visualization.action_visualizer import ActionVisualizer
            return ActionVisualizer()
        else:
            msg = "model_name is {}, but not implemented".format(
                visualizer_name)
            raise NotImplementedError(msg)
