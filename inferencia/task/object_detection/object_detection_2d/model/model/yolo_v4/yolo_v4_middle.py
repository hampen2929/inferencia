from .yolo_v4 import YoloV4


class YoloV4Middle(YoloV4):

    model_detail_name = "YoloV4Middle"
    input_width = 416
    input_height = 416
    weight_loc = 'google_drive'
    weight_url = "1SFW44cFCpgRzVPNYAdDauaJcF4w7ufG8"

    def __init__(self,
                 model_path,
                 model_precision,
                 conf_thresh,
                 nms_thresh,
                 label_name):
        super().__init__(model_path=model_path,
                         model_precision=model_precision,
                         conf_thresh=conf_thresh,
                         nms_thresh=nms_thresh,
                         label_name=label_name)
