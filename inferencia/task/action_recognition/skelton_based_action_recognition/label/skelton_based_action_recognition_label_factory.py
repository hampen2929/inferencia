from ..label.skelton_based_action_recognition_label_name import SkeltonBasedActionRecognitionName


class SkeltonBasedActionRecognitionLabelFactory():
    def create(label_name):
        if SkeltonBasedActionRecognitionName.golf_swing.value == label_name:
            from ..label.label.golfl_action_label import GolfSwingLabel
            return GolfSwingLabel()
        else:
            raise NotImplementedError(
                "{} is not implemented.".format(label_name))
