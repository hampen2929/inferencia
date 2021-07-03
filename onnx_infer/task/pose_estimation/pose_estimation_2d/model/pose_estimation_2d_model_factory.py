from .pose_estimation_2d_model_name import PoseEstimationModelName


class PoseEstimation2DFactory():
    def create(model_name="MoveNet-Thunder",
               model_path=None,
               model_precision="FP32"):
        if model_name == PoseEstimationModelName.move_net_thunder.value:
            from .move_net.move_net_thunder import MoveNetThunder
            return MoveNetThunder(model_path=model_path,
                                  model_precision=model_precision)

        elif model_name == PoseEstimationModelName.move_net_lightning.value:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)

        else:
            msg = "model_name is {}, but not implemented".format(model_name)
            raise NotImplementedError(msg)
