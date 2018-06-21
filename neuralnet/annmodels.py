from config import Column
from keras.layers import Dense
from neuralnet.models import NetModel
# from neuralnet.annmodels import BaselineModel

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
