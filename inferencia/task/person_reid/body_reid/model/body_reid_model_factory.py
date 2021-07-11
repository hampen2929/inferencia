from .body_reid_model_name import BodyReidModelName


class BodyReidModelFactory():
    def create(model_name,
               model_path,
               model_precision):
        if model_name == BodyReidModelName.osnet_x0_25.value:
            from .model.osnet.osnet_x0_25 import OSNetX025
            return OSNetX025(model_path,
                             model_precision)
        else:
            msg = "{} is not implemented. Choose from {}.".format(
                model_name,
                BodyReidModelName.names()
            )
            raise NotImplementedError(msg)
