from config import Column
from keras.layers import Dense
from collections import namedtuple


DataIO = namedtuple('DataIO', 'input output')


def io_split(df, input_cols, output_cols):
    """ Take a dataframe and split it into input and output columns.
        Currently just hardcoding split by columns
    """
    return


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


class BaselineModel(NetModel):
    INPUT_COLS = [
        Column.TIME, Column.INLET_FLOWRATE,
        Column.INLET_TDS
    ]
    OUTPUT_COLS = [
        Column.PERM_FLOWRATE,
        Column.PERM_COND,
    ]

    def __init__(self, input_cols=INPUT_COLS,
                 output_cols=OUTPUT_COLS):
        super().__init__(input_cols, output_cols)

    def make_net(self, model):
        self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(Dense(output_dim=6, init='uniform',
                        activation='relu'))
        self._add_output_layer(model)


class AllOutputsModel(BaselineModel):
    OUTPUT_COLS = [
        Column.PERM_FLOWRATE,
        Column.PERM_COND,
        Column.MEMBR_REJ_FLOWRATE,
        Column.MEMBR_REJ_TDS,
        Column.TANK_TDS
    ]

    def __init__(self, input_cols=BaselineModel.INPUT_COLS,
                 output_cols=OUTPUT_COLS):
        super().__init__(input_cols, output_cols)


class RejectFlowrate(AllOutputsModel):
    OUTPUT_COLS = [
        Column.MEMBR_REJ_FLOWRATE,
    ]

    def __init__(self):
        super().__init__(output_cols=RejectFlowrate.OUTPUT_COLS)

    def make_net(self, model):
        self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(Dense(output_dim=15, init='uniform',
                        activation='relu'))
        model.add(Dense(output_dim=4, init='uniform',
                        activation='relu'))
        self._add_output_layer(model)


class RejectTDS(AllOutputsModel):
    OUTPUT_COLS = [
        Column.MEMBR_REJ_TDS,
    ]

    def __init__(self):
        super().__init__(output_cols=RejectTDS.OUTPUT_COLS)

    def make_net(self, model):
        self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(Dense(output_dim=6, init='uniform',
                        activation='relu'))
        model.add(Dense(output_dim=4, init='uniform',
                        activation='relu'))
        self._add_output_layer(model)


class TankTDS(AllOutputsModel):
    OUTPUT_COLS = [
        Column.TANK_TDS,
    ]

    def __init__(self):
        super().__init__(output_cols=TankTDS.OUTPUT_COLS)

    def make_net(self, model):
        self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(Dense(output_dim=6, init='uniform',
                        activation='sigmoid'))
        model.add(Dense(output_dim=4, init='uniform',
                        activation='sigmoid'))
        self._add_output_layer(model)


MODELS = (
    BaselineModel, AllOutputsModel, RejectFlowrate, RejectTDS, TankTDS
)


def make_model(model_name):
    """ Factory function for the model, based on the model
        name
    """
    for ModelCls in MODELS:
        if ModelCls.__name__.lower() == model_name.lower():
            return ModelCls()
    raise ValueError("Invalid model name: " + model_name)
