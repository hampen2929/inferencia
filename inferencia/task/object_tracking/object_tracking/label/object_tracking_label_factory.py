from ..label.template_label_name import ObjectTrackingLabelName
from inferencia.util.label.label_process import to_dict


class ObjectTrackingLabelFactory():
    def create(label_name):
        if ObjectTrackingLabelName.mot.value == label_name:
            from ..label.label.coco_label import COCOLabel
            # label_dict = to_dict(COCOLabel)
            return COCOLabel()

        elif TemplateLabelName.pascalvoc.value == label_name:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))

        else:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))
