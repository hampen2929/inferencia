from .label.tmp_label_factory import TmpLabelFactory
from .model.tmp_model_factory import TmpModelFactory
from .visualization.tmp_visualizer_factory import TmpVisualizerFactory


class TmpManager():

    def get_model(model_name="Tmp",
                  model_path=None):
        return TmpModelFactory.create(model_name,
                                      model_path)

    def get_visualizer(visualizer_name="Visualizer"):
        return TmpVisualizerFactory.create(visualizer_name)

    def get_label(label_name):
        return TmpLabelFactory.create(label_name)
