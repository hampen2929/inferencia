import os.path as osp
from typing import List, Tuple, Union

import numpy as np
import cv2
import onnxruntime as ort

from ...pose_estimation_2d_model import PoseEstimation2dModel
from ...pose_estimation_2d_result import PoseEstimation2dResult

from .post_process import get_final_preds

from inferencia.util.file.file import get_model_path
from inferencia.util.file.file import download_from_google_drive
from inferencia.util.pre_process.normalize import normalize
from inferencia.util.pre_process.resize import resize
from inferencia.util.pre_process.validate import validate_image


class TransPoseModel(PoseEstimation2dModel):
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]

    def __init__(self,
                 model_path: str,
                 model_precision: str,
                 test_blur_kernel: int = 3
                 ):
        self.sess = ort.InferenceSession(model_path)
        _, _, in_h, in_w = self.sess.get_inputs()[0].shape
        self.in_size = (in_h, in_w)
        self.test_blur_kernel = test_blur_kernel

    def pre_process(self, images: Union[np.ndarray, List]) -> np.ndarray:
        """Resize, normalize and transform for forward input

        Args:
            images (Union[np.ndarray, List]): single image as np.ndarray. single or multi image as list.

        Returns:
            np.ndarray: pre_processed [b, c, h, w] np.ndarray image
        """
        images = validate_image(images)
        image_sizes = []
        image_arr = []
        for image in images:
            image_sizes.append(image.shape)
            image = resize(image,
                           height=self.in_size[0],
                           width=self.in_size[1],)
            image = normalize(image)
            image_arr.append(image)
        image_arr = np.array(image_arr)
        return image_arr, image_sizes

    def forward(self, pre_processed: np.ndarray) -> List[np.ndarray]:
        """[summary]

        Args:
            pre_processed (np.ndarray): pre_processd data

        Returns:
            List[np.ndarary]: forwarded result as list like heatmap of each key point
            np.ndarray size is (b, 17, 64, 48)
            17 means key points
        """
        fwd_rets = self.sess.run(None,
                                 {'actual_input_1': pre_processed.astype(np.float32)})
        return fwd_rets

    def post_process(self,
                     fwd_rets: List,
                     image_sizes: List[Tuple],
                     ) -> List[PoseEstimation2dResult]:
        """pose post processed

        Args:
            fwd_rets (List): [description]
            image_size (Tuple): [description]

        Returns:
            List[PoseEstimation2dResult]: [description]
        """
        query_locations = []
        post_processed = []
        for fwd_ret, image_size in zip(fwd_rets, image_sizes):
            pose_ret = np.zeros((17, 3))
            preds, maxvals = get_final_preds(fwd_ret,
                                             None,
                                             None,
                                             transform_back=False,
                                             test_blur_kernel=self.test_blur_kernel)

            pose_x = (preds[0][:, 0] * image_size[1] / self.in_size[1]) * 4
            pose_y = (preds[0][:, 1] * image_size[0] / self.in_size[0]) * 4
            confidence = maxvals[0][:, 0]

            pose_ret[:, 0] = pose_x
            pose_ret[:, 1] = pose_y
            pose_ret[:, 2] = confidence

            query_location = np.array([p*4+0.5 for p in preds[0]])

            post_processed.append(PoseEstimation2dResult(pose=pose_ret,
                                                         image_height=image_size[0],
                                                         image_width=image_size[1],
                                                         pose_norm=None,
                                                         outputs=None,
                                                         heatmap=fwd_ret,
                                                         query_location=query_location,
                                                         ))
        return post_processed

    def inference(self, images: Union[np.ndarray, List]) -> List:
        # return super().inference(images)
        pre_processed, image_sizes = self.pre_process(images)
        forwarded = self.forward(pre_processed)
        post_processed = self.post_process(forwarded, image_sizes)
        return post_processed
