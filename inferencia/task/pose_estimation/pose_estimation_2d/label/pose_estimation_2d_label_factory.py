from ..label.pose_estimation_2d_label_name import PoseEstimation2DLabelName


class PoseEstimation2DLabelFactory():
    def create(label_name):
        if PoseEstimation2DLabelName.coco_keypoint_label.value == label_name:
            from ..label.label.coco_keypoint_label import COCOKeypointLabel
            return COCOKeypointLabel

        else:
            raise NotImplementedError(
                "{} is not implemented. {} can be chosen.".format(label_name, [e.value for e in PoseEstimation2DLabelName]
                                                                  ))
