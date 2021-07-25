from enum import Enum


class PoseEstimationModelName(Enum):
    move_net_thunder = "MoveNet-Thunder"
    move_net_lightning = "MoveNet-Lightning"
    trans_pose = "TransPose"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in PoseEstimationModelName}

    @staticmethod
    def names():
        return [l.name for l in PoseEstimationModelName]

    @staticmethod
    def values():
        return [l.value for l in PoseEstimationModelName]
