import pickle
import numpy as np
import pandas as pd

from ...skelton_based_action_recognition_model import SkeltonBasedActionRecognitionModel
from ...skelton_based_action_recognition_result import SkeltonBasedActionRecognitionResult


class LightGBM(SkeltonBasedActionRecognitionModel):

    def __init__(self,
                 model_path,
                 label_dict):
        self.model = self.get_model(model_path)
        self.label_dict = label_dict

    def set_model(self, model_path):
        return pickle.load(open(model_path, 'rb'))

    def pre_process(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError()
        return data

    def forward(self, data):
        confidences = self.model.predict(data)
        return confidences

    def post_process(self, confidences):
        class_id = np.argmax(confidences, axis=1)[0]
        confidence = confidences[class_id]
        class_name = self.label_dict[class_id]
        skelton_based_action_recognition_result = SkeltonBasedActionRecognitionResult(class_id,
                                                                                      class_name,
                                                                                      confidence,
                                                                                      confidences)
        return skelton_based_action_recognition_result

    def inference(self, data):
        pre_processed = self.pre_process(data)
        confidences = self.forward(pre_processed)
        act_rec_ret = self.post_process(confidences)
        return act_rec_ret
