from ..label.body_reid_label_name import BodyReidLabelName


class BodyReidLabelFactory():
    def create(label_name):
        if BodyReidLabelName.body_reid.value == label_name:
            from ..label.label.body_reid_label import BodyReidLabel
            return BodyReidLabel()
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                label_name,
                [i for i in BodyReidLabelName]
            )
            raise NotImplementedError(msg)
