from inferencia.util.file import download_from_google_drive
import os.path as osp
from typing import Union

import numpy as np
import cv2
import onnxruntime

from ...pose_estimation_2d_model import PoseEstimation2dModel
from ...pose_estimation_2d_result import PoseEstimation2dResult

from .process import change_ratio
from .......util.file import get_model_path


class MoveNet(PoseEstimation2dModel):
    in_w = None
    in_h = None
    task_major_name = 'pose_estimation'
    task_minor_name = 'pose_estimation_2d'
    model_name = 'MoveNet'
    model_detail_name = None
    weight_url = None
    change_ratio = 16 / 9

    def __init__(self,
                 model_path,
                 model_precision):
        if model_path is None:
            model_path = get_model_path(self.task_major_name,
                                        self.task_minor_name,
                                        self.model_name,
                                        self.model_detail_name,
                                        model_precision)

        if not osp.exists(model_path):
            download_from_google_drive(self.weight_url, model_path)

        self.sess = self.get_inference_session(model_path)

    def get_inference_session(self, model_path):
        return onnxruntime.InferenceSession(model_path)

    def inference(self, images: Union[np.ndarray, list]) -> list:
        print(images.shape)
        pre_process_results, height, width = self.pre_process(images)
        print(height, width)
        forward_results = self.forward(pre_process_results)
        return self.post_process(forward_results, height, width)

    def pre_process(self, images):
        if len(images.shape) == 3:
            height, width, _ = images.shape

        elif len(images.shape) == 4:
            _, height, width, _ = images.shape

        # 人を検出した縦長の矩形だと姿勢推定がうまくいかないことがあるので横長の比率に変更
        images, height, width = change_ratio(images,
                                             height,
                                             width,
                                             change_ratio=self.change_ratio)
        frame = cv2.resize(images, (self.in_w, self.in_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)

        frame = frame.astype('float32')

        pre_process_results = {self.sess.get_inputs()[0].name: frame}
        return pre_process_results, height, width

    def forward(self, pre_process_results):
        forward_results = self.sess.run(None, pre_process_results)
        return forward_results

    def post_process(self,
                     forward_results,
                     image_height,
                     image_width,
                     ):
        pose_est_rets = []
        for outputs in forward_results:
            pose_norm = outputs[0][0]  # (y, x, conf)
            pose = np.zeros(pose_norm.shape)

            pose[:, 0] = ((pose_norm[:, 1] * image_width)).round()
            pose[:, 1] = ((pose_norm[:, 0] * image_height)).round()
            pose[:, 2] = pose_norm[:, 2]
            pose_est_ret = PoseEstimation2dResult(pose=pose,
                                                  pose_norm=pose_norm,
                                                  outputs=outputs,
                                                  image_height=image_height,
                                                  image_width=image_width
                                                  )

            pose_est_rets.append(pose_est_ret)
        return pose_est_rets
