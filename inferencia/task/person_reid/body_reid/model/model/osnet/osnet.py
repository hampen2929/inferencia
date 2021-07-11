import os.path as osp
from typing import List, Union

import numpy as np
import onnxruntime as ort

from ...body_reid_model import BodyReidModel
from ...body_reid_result import BodyReidResult

from inferencia.util.pre_process.validate import validate_image
from inferencia.util.pre_process.resize import resize
from inferencia.util.pre_process.normalize import normalize
from inferencia.util.file.file import get_model_path, download_from_google_drive
from inferencia.util.logger.logger import Logger


class OSNet(BodyReidModel):
    task_major_name = "PersonReid"
    task_minor_name = "BodyReid"
    model_name = "OSNet"
    model_detail_name = "osnet_x0_25"
    model_precision = "FP32"

    input_width = None
    input_height = None
    weight_url = None

    norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    input_height = 256
    input_width = 128

    def __init__(self,
                 model_path,
                 model_precision):
        self.logger = Logger(__class__.__name__)
        init_msg = "\n===================== \n Initialize {}-{}-{} \n=====================\n"
        init_msg = init_msg.format(self.task_minor_name,
                                   self.model_name,
                                   self.model_detail_name)
        self.logger.info(init_msg)

        model_path = self.get_model_path(model_path,
                                         self.task_major_name,
                                         self.task_minor_name,
                                         self.model_name,
                                         self.model_detail_name,
                                         model_precision)
        self.download_model(self.weight_url, model_path)

        # TODO: OSNetFactory
        # if model_name == BodyReidModelName.osnet_x0_25:
        self.sess = ort.InferenceSession(model_path)

    def get_model_path(self,
                       model_path,
                       task_major_name,
                       task_minor_name,
                       model_name,
                       model_detail_name,
                       model_precision):
        if model_path is None:
            model_path = get_model_path(task_major_name,
                                        task_minor_name,
                                        model_name,
                                        model_detail_name,
                                        model_precision)
        else:
            pass
        return model_path

    def download_model(self, weight_url, model_path):
        if not osp.exists(model_path):
            download_from_google_drive(weight_url, model_path)
            msg = "download weight from {weight_url} and save to {model_path}".format(weight_url=weight_url,
                                                                                      model_path=model_path)
            self.logger.info(msg)

    def pre_process(self, image: Union[List, np.ndarray]) -> np.ndarray:
        images = validate_image(image)
        pre_processed = []
        for image in images:
            image = resize(image, self.input_height, self.input_width)
            image = normalize(image)
            pre_processed.append(image)
        pre_processed = np.array(pre_processed)
        return pre_processed

    def forward(self, pre_processed: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            pre_processed (np.ndarray): [b, c, h, w]

        Returns:
            np.ndarray: [b, dim]]
        """

        ort_inputs = {self.sess.get_inputs()[0].name: pre_processed}
        ort_outs = self.sess.run(None, ort_inputs)[0]
        return ort_outs

    def post_process(self, fwd_rets):
        return [BodyReidResult(feature=fwd_ret) for fwd_ret in fwd_rets]

    def inference(self, image: Union[List, np.ndarray]):
        pre_processed = self.pre_process(image)
        fwd_rets = self.forward(pre_processed)
        body_reid_rets = self.post_process(fwd_rets)
        return body_reid_rets
