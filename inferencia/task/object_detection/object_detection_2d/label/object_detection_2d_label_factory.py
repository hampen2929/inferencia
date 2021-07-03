from ..label.object_detection_2d_label_name import ObjectDetection2DLabelNames


class ObjectDetection2DLabelFactory():
    def create(label_name):
        if ObjectDetection2DLabelNames.coco.value == label_name:
            from ..label.coco_label import COCOLabel
            return COCOLabel

        elif ObjectDetection2DLabelNames.pascalvoc.value == label_name:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))

        else:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))
