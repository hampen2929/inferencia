from .pose_estimation_2d_visualizer_name import PoseEstimation2DVisualizerName
from ..label.pose_estimation_2d_label_factory import PoseEstimation2DLabelFactory


class PoseEstimation2DVisualizerFactory():
    def create(visualizer_name="PoseVisualizer",
               label_name="COCOKeyPointLabel"):
        if visualizer_name == PoseEstimation2DVisualizerName.pose_visualizer.value:
            from .visualization.pose_vilualizer import PoseVilualizer

            pose_label = PoseEstimation2DLabelFactory.create(
                label_name=label_name)
            pose_visualizer = PoseVilualizer(body_edges=pose_label.body_edges)
            return pose_visualizer
        else:
            msg = "model_name is {}, but not implemented".format(
                visualizer_name)
            raise NotImplementedError(msg)
