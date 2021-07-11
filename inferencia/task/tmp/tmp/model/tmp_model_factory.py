from .tmp_model_name import TmpModelName


class TmpModelFactory():
    def create(model_name,
               model_path):
        if model_name == TmpModelName.tmp.value:
            from .model import Tmp
            return Tmp(model_path=model_path)
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                model_name,
                TmpModelName.names
            )
            raise NotImplementedError(msg)
