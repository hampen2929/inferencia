from ..label.tmp_label_name import TmpLabelName


class TmpLabelFactory():
    def create(label_name):
        if TmpLabelName.tmp.value == label_name:
            from ..label.label.tmp_label import TmpLabel
            return TmpLabel()
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                label_name,
                [i for i in TmpLabelName]
            )
            raise NotImplementedError(msg)
