from preprocessing.ro_data import Column
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

    def make_net(self, model):
        pass

    def io_split(self, df):
        """ Split df into np arrays of input and output """
        return DataIO(df.loc[:, self.input_cols],
                      df.loc[:, self.output_cols])


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
        # Adding the input layer and the first hidden layer
        model.add(Dense(output_dim=6, init='uniform',
                        activation='relu', input_dim=len(self.input_cols)))
        # Adding the second hidden layer
        model.add(Dense(output_dim=6, init='uniform',
                        activation='relu'))
        # Adding the output layer
        model.add(Dense(output_dim=len(self.output_cols), init='uniform',
                        activation='sigmoid'))


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


MODELS = (
    ("Baseline", BaselineModel),
    ("AllOutput", AllOutputsModel),
)


def make_model(model_name):
    """ Factory function for the model, based on the model
        name
    """
    for name, ModelCls in MODELS:
        if name.lower() == model_name.lower():
            return ModelCls()
    return None
