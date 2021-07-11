from typing import List
import numpy as np
import cv2

from inferencia.task.person_reid.body_reid.body_reid_manager import BodyReidManager
from inferencia.task.person_reid.body_reid.model.body_reid_result import BodyReidResult

TEST_DOG_IMG = './data/dog.jpg'


class TestBodyReidManager:

    def test_inference(self):
        img = cv2.imread(TEST_DOG_IMG)
        osnet_model = BodyReidManager.get_model()
        body_reid_results = osnet_model.inference(img)
        assert isinstance(body_reid_results, List)
        for body_reid_result in body_reid_results:
            assert isinstance(body_reid_result, BodyReidResult)
            assert isinstance(body_reid_result.feature, np.ndarray)
            assert body_reid_result.feature.shape == (512,)

    def test_compare_pytorch_model(self):
        pass
