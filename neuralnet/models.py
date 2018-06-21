from config import Column
from keras.layers import Dense
from collections import namedtuple

DataIO = namedtuple('DataIO', 'input output')


class NetModel(object):
    """ The neural net model object, specified the
        arch, input and output of the net.
    """
    def __init__(self, input, output):
        self.input_cols, self.output_cols = input, output
        self.input_act, self.output_act = 'relu', 'sigmoid'

    def _add_input_layer(self, model):
        model.add(Dense(output_dim=6, init='uniform',
                        activation=self.input_act,
                        input_dim=len(self.input_cols)))

    def _add_output_layer(self, model):
        model.add(Dense(output_dim=len(self.output_cols), init='uniform',
                        activation=self.output_act))

    def make_net(self, model):
        self._add_input_layer(model)
        self._add_output_layer(model)

    def io_split(self, df):
        """ Split df into np arrays of input and output """
        return DataIO(df.loc[:, self.input_cols].values,
                      df.loc[:, self.output_cols].values)

    def post_process(self, ts_data, predicted):
        return predicted



def make_model(model_name):
    """ Factory function for the model, based on the model
        name
    """
    from neuralnet.rnnmodels import RnnBaseline, TankTdsFinal
    from neuralnet.annmodels import BaselineModel, \
            AllOutputsModel, RejectFlowrate, RejectTDS, TankTDS
    MODELS = (
        BaselineModel, AllOutputsModel, RejectFlowrate, RejectTDS, TankTDS, 
        RnnBaseline, TankTdsFinal
    )
    for ModelCls in MODELS:
        if ModelCls.__name__.lower() == model_name.lower():
            return ModelCls()
    raise ValueError("Invalid model name: " + model_name)
