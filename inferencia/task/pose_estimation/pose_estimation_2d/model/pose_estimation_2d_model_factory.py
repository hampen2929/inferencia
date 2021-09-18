from .pose_estimation_2d_model_name import PoseEstimationModelName


class PoseEstimation2DModelFactory:
    def create(model_name="MoveNet-Thunder", model_path=None, model_precision="FP32"):
        if model_name == PoseEstimationModelName.move_net_thunder.value:
            from .model.move_net.move_net_thunder import MoveNetThunder

            return MoveNetThunder(
                model_path=model_path, model_precision=model_precision
            )

        elif model_name == PoseEstimationModelName.trans_pose.value:
            from .model.trans_pose.trans_pose_model import TransPoseModel

            return TransPoseModel(
                model_path=model_path, model_precision=model_precision
            )

        else:
            msg = f"{PoseEstimationModelName.values()} are supported. {model_name} is not implemented."
            raise NotImplementedError(msg)
