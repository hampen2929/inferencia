import pytest
import cv2

from inferencia.task.pose_estimation.pose_estimation_2d.pose_estimation_2d_manager import PoseEstimation2DManager
from inferencia.task.pose_estimation.pose_estimation_2d.model.pose_estimation_2d_result import PoseEstimation2dResult
from inferencia.task.pose_estimation.pose_estimation_2d.model.pose_estimation_2d_model_name import PoseEstimationModelName

TEST_DOG_IMG = './data/person_image.png'
MODEL_NAMES = PoseEstimationModelName.values()


class TestPoseEstimation2DManager():

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_get_model(self, model_name):
        img = cv2.imread(TEST_DOG_IMG)
        pose_2d_model = PoseEstimation2DManager.get_model(model_name)
        pose_rets = pose_2d_model.inference(img)
        for pose_ret in pose_rets:
            assert isinstance(pose_ret, PoseEstimation2dResult)

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_output_accuracy(self, model_name):
        # TODO: inference and calc pose accuracy

        # img = cv2.imread(TEST_DOG_IMG)
        # pose_2d_model = PoseEstimation2DManager.get_model(model_name)
        # pose_rets = pose_2d_model.inference(img)
        # for pose_ret in pose_rets:
        #     assert isinstance(pose_ret, PoseEstimation2dResult)

        pass
