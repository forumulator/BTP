from config import Column
from keras.layers import Dense, LSTM
from collections import namedtuple
from neuralnet.models import NetModel

DataIO = namedtuple('DataIO', 'input output')

class RnnBaselineModel(NetModel):
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

    def make_net(self, model):
        self._add_input_layer(model)
        # Adding the second hidden layer
        model.add(LSTM(output_dim=6, init='uniform',
                        activation='relu'))
        self._add_output_layer(model)
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')