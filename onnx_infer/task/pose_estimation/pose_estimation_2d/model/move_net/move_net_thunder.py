
from .move_net import MoveNet


class MoveNetThunder(MoveNet):

    in_w = 256
    in_h = 256
    model_detail_name = "Thunder"
    weight_loc = 'google_drive'
    weight_url = "1O2d9zDOddB_MfuXG7mc1sj7Y18UmT4QR"

    def __init__(self,
                 model_path,
                 model_precision):
        super().__init__(model_path=model_path,
                         model_precision=model_precision)
