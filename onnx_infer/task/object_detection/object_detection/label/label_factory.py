from ..label.label_names import LabelNames


class LabelFactory():
    def create(label_name):
        if LabelNames.coco.value == label_name:
            from ..label.coco_label import COCOLabel
            return COCOLabel

        elif LabelNames.pascalvoc.value == label_name:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))

        else:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))
