import cv2
import numpy as np
from copy import deepcopy

import torch
import torchvision.transforms as T

from inferencia.task.pose_estimation.pose_estimation_2d.model.pose_estimation_2d_result import PoseEstimation2dResult
from inferencia.task.pose_estimation.pose_estimation_2d.model.model.trans_pose.trans_pose_model import TransPoseModel

from inferencia.util.pre_process.resize import resize
from inferencia.util.pre_process.normalize import normalize

TEST_DOG_IMG = './data/person_image.png'


class TestTransPoseModel():

    def test_inference(self):
        img = cv2.imread(TEST_DOG_IMG)
        model = TransPoseModel(model_path=None,
                               model_precision="FP32")
        pose_rets = model.inference(img)
        for pose_ret in pose_rets:
            assert isinstance(pose_ret, PoseEstimation2dResult)

    def test_compare_onnx_and_pytorch_output(self):
        img = cv2.imread(TEST_DOG_IMG)
        # Torch
        img_torch = img.copy()

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        normalize_torch = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        trns = T.Compose([
            T.ToTensor(),
            normalize_torch,
        ])

        model_torch = torch.hub.load('yangsenius/TransPose:main',
                                     'tpr_a4_256x192',
                                     pretrained=True)
        model_torch.to(device)
        with torch.no_grad():
            model_torch.eval()
            # img_torch = cv2.resize(img_torch, dsize=(192, 256))
            img_torch = resize(img_torch, width=192, height=256)
            img_torch = trns(img_torch)
            inputs_torch = torch.cat([img_torch.to(device)]).unsqueeze(0)
            outputs_torch = model_torch(inputs_torch)
        outputs_torch = outputs_torch.detach().cpu().numpy()

        # ONNX
        img_onnx = img.copy()
        model_onnx = TransPoseModel(model_path=None,
                                    model_precision="FP32")
        pre_processed_onnx, _ = model_onnx.pre_process(img_onnx)
        outputs_onnx = model_onnx.forward(pre_processed_onnx)

        assert outputs_torch.shape == outputs_onnx[0].shape

        # np.testing.assert_almost_equal(outputs_torch,
        #                                outputs_onnx[0])

        np.testing.assert_allclose(outputs_torch,
                                   outputs_onnx[0],
                                   rtol=1e-03,
                                   atol=1e-05)
