import pytest

from inferencia.task.pose_estimation.pose_estimation_2d.model.pose_estimation_2d_model_factory import PoseEstimation2DModelFactory
from inferencia.task.pose_estimation.pose_estimation_2d.model.model.move_net.move_net_thunder import MoveNetThunder
from inferencia.task.pose_estimation.pose_estimation_2d.model.model.trans_pose.trans_pose_model import TransPoseModel


class TestPoseEstimation2DModelFactory():

    @pytest.mark.parametrize(
        'model_name, model_class', [
            ("MoveNet-Thunder", MoveNetThunder),
            ("TransPose", TransPoseModel)
        ]

    )
    def test_create(self, model_name, model_class):
        model = PoseEstimation2DModelFactory.create(model_name)
        assert isinstance(model, model_class)
