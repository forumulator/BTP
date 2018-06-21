from config import Column
from keras.layers import Dense, LSTM
from collections import namedtuple
from neuralnet.models import NetModel, DataIO
from neuralnet.annmodels import BaselineModel

DataIO = namedtuple('DataIO', 'input output')

class TankTdsFinal(NetModel):
    INPUT_COLS = [
        Column.TIME, Column.INLET_FLOWRATE,
        Column.INLET_TDS
    ]
    OUTPUT_COLS = [
        Column.TANK_TDS,
    ]

    def __init__(self, input_cols=INPUT_COLS,
                 output_cols=OUTPUT_COLS):
        super().__init__(input_cols, output_cols)

    def io_split(self, df):
        """ Split and reshape the output for an RNN """
        inp, out = df.loc[:, self.input_cols].values, \
            df.loc[:, self.output_cols].values
        inp = inp.reshape((inp.shape[0], 1, inp.shape[1]))
        # out = out.reshape((out.shape[0], 1, out.shape[1]))
        return DataIO(inp, out)

    def make_net(self, model):
        # self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(LSTM(50, input_shape=(1, len(self.input_cols))))
        model.add(Dense(1))
        # self._add_output_layer(model)
        # model = Sequential()
        # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        # model.add(Dense(1))
        # model.compile(loss='mae', optimizer='adam')

class RnnBaseline(BaselineModel):
    OUTPUT_COLS = [
        Column.TANK_TDS
    ]

    def __init__(self, input_cols=BaselineModel.INPUT_COLS,
                 output_cols=OUTPUT_COLS):
        super().__init__(input_cols, output_cols)