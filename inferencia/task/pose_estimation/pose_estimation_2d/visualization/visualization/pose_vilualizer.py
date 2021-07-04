from ..pose_estimation_2d_vilualizer import PoseEstimation2DVilualizer
from .visualize_pose import visualize_pose


class PoseVilualizer(PoseEstimation2DVilualizer):
    def __init__(self,
                 body_edges):
        self.body_edges = body_edges

    def visualize(self,
                  image,
                  pose):
        pose_image = visualize_pose(image.copy(),
                                    pose,
                                    body_edges=self.body_edges)
        return pose_image
