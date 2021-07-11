import os.path as osp
from typing import Union

import numpy as np
import onnxruntime

from .process import (pre_process,
                      post_processing,
                      validate_image,
                      validate_bbox)

from ...object_detection_2d_model import ObjectDetection2DModel
from ...object_detection_2d_result import ObjectDetection2DResult
from ....label.object_detection_2d_label_factory import ObjectDetection2DLabelFactory

from .......util.file.file import get_model_path, download_from_google_drive
from .......util.logger.logger import Logger


class YoloV4(ObjectDetection2DModel):
    task_major_name = "ObjectDetection"
    task_minor_name = "ObjectDetection2D"
    model_name = "YoloV4"
    model_detail_name = None
    input_width = None
    input_height = None
    weight_url = None

    def __init__(self,
                 model_path,
                 model_precision,
                 conf_thresh,
                 nms_thresh,
                 label_name):
        self.logger = Logger(__class__.__name__)
        init_msg = "\n===================== \n Initialize {}-{}-{} \n=====================\n".format(self.task_minor_name,
                                                                                                     self.model_name,
                                                                                                     self.model_detail_name)
        self.logger.info(init_msg)

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        model_path = self.get_model_path(model_path,
                                         self.task_major_name,
                                         self.task_minor_name,
                                         self.model_name,
                                         self.model_detail_name,
                                         model_precision)
        self.download_model(self.weight_url, model_path)
        self.sess = self.get_inference_session(model_path)

        # First inference is too slow. Should be done here.
        dummy_image = self.get_dummy_image()
        self.inference(dummy_image)
        self.logger.info("Initial inference")

        label = ObjectDetection2DLabelFactory.create(label_name)
        self.label_dict = label.to_json()

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

    def get_inference_session(self, model_path):
        return onnxruntime.InferenceSession(model_path)

    def inference(self,
                  images: Union[np.ndarray, list]) -> list:
        pre_proc_rets, image_sizes = self.pre_process(images)
        fwd_rets = self.forward(pre_proc_rets)
        post_proc_rets = self.post_process(fwd_rets, image_sizes)
        return post_proc_rets

    def pre_process(self, images):
        images = validate_image(images)
        pre_proc_rets, image_sizes = pre_process(images,
                                                 self.input_width,
                                                 self.input_height)
        return pre_proc_rets, image_sizes

    def forward(self, images):
        output = self.sess.run(None, {self.sess.get_inputs()[0].name: images})
        return output

    def post_process(self, fwd_rets, image_sizes):
        frames_boxes = post_processing(fwd_rets,
                                       self.conf_thresh,
                                       self.nms_thresh)
        obj_det_rets = []
        for frame_boxes, image_size in zip(frames_boxes, image_sizes):
            for box in frame_boxes:
                image_height, image_width, _ = image_size
                xmin, ymin, xmax, ymax, confidence, _, class_id = box
                xmin, ymin, xmax, ymax = validate_bbox(xmin,
                                                       ymin,
                                                       xmax,
                                                       ymax,
                                                       image_height,
                                                       image_width)
                obj_det_ret = ObjectDetection2DResult(class_id,
                                                      self.label_dict[class_id],
                                                      xmin,
                                                      ymin,
                                                      xmax,
                                                      ymax,
                                                      confidence)
                obj_det_rets.append(obj_det_ret)
        return obj_det_rets

    def get_dummy_image(self):
        input_shape = self.sess.get_inputs()[0].shape
        _, _, self.input_height, self.input_width = input_shape
        dummy_image = np.zeros((self.input_height, self.input_width, 3),
                               dtype=np.uint8)
        return dummy_image
