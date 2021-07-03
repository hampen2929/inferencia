from .visualize_pose import visualize_pose


class PoseVilualizer():
    def __init__(self,
                 body_edges):
        self.body_edges = body_edges

    def visualize(self,
                  image,
                  pose):
        pose_image = visualize_pose(image,
                                    pose,
                                    body_edges=self.body_edges)
        return pose_image
