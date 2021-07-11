from .label.body_reid_label_factory import BodyReidLabelFactory
from .model.body_reid_model_factory import BodyReidModelFactory
from .visualization.body_reid_visualizer_factory import BodyReidVisualizerFactory


class BodyReidManager():

    def get_model(model_name="osnet_x0_25",
                  model_path=None,
                  model_precision="FP32"):
        return BodyReidModelFactory.create(model_name,
                                           model_path,
                                           model_precision)

    def get_visualizer(visualizer_name="Visualizer"):
        return BodyReidVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return BodyReidLabelFactory.create(label_name)
