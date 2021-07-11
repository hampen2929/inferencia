from .osnet import OSNet


class OSNetX025(OSNet):
    model_detail_name = "OSNetX025"
    input_height = 256
    input_width = 128
    weight_loc = 'google_drive'
    weight_url = ""

    def __init__(self,
                 model_path,
                 model_precision):
        model_path = "/workspace/notebook/sandbox/deep-person-reid/osnet_x0_25.onnx"
        super().__init__(model_path=model_path,
                         model_precision=model_precision)
