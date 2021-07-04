from .label.pose_estimation_2d_label_factory import PoseEstimation2DLabelFactory
from .model.pose_estimation_2d_model_factory import PoseEstimation2DModelFactory
from .visualization.pose_estimation_2d_visualizer_factory import PoseEstimation2DVisualizerFactory


class PoseEstimation2DManager():

    def get_model(model_name):
        return PoseEstimation2DModelFactory.create(model_name=model_name)

    def get_visualizer(visualizer_name="PoseVisualizer",
                       label_name="COCOKeyPointLabel"):
        return PoseEstimation2DVisualizerFactory.create(visualizer_name, label_name)

    def get_label(label_name):
        return PoseEstimation2DLabelFactory.create(label_name)
