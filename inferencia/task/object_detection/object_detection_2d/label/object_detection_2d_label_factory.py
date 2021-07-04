from ..label.object_detection_2d_label_name import ObjectDetection2DLabelName
from inferencia.util.label.label_process import to_dict


class ObjectDetection2DLabelFactory():
    def create(label_name):
        if ObjectDetection2DLabelName.coco.value == label_name:
            from ..label.label.coco_label import COCOLabel
            # label_dict = to_dict(COCOLabel)
            return COCOLabel()

        elif ObjectDetection2DLabelName.pascalvoc.value == label_name:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))

        else:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))
