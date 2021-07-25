import pytest
import cv2

from inferencia.task.pose_estimation.pose_estimation_2d.pose_estimation_2d_manager import PoseEstimation2DManager
from inferencia.task.pose_estimation.pose_estimation_2d.model.pose_estimation_2d_result import PoseEstimation2dResult


TEST_DOG_IMG = './data/person_image.png'


class TestPoseEstimation2DManager():

    @pytest.mark.parametrize('model_name', [
        "MoveNet-Thunder",
        "TransPose",
    ])
    def test_get_model(self, model_name):
        img = cv2.imread(TEST_DOG_IMG)
        pose_2d_model = PoseEstimation2DManager.get_model(model_name)
        pose_rets = pose_2d_model.inference(img)
        for pose_ret in pose_rets:
            assert isinstance(pose_ret, PoseEstimation2dResult)
