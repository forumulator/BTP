from collections import namedtuple
from preprocessing.ro_data import Column
from sklearn.model_selection import train_test_split


class Splitter(object):
    def __init__(self, ts_size):
        self.ts_size = ts_size

    def split(self, df):
        """ Return a two tuple of tr, ts data """
        tr, ts = train_test_split(df, test_size=self.ts_size)
        return tr, ts


def train_neural_net(df, args=None):
    pass


DataIO = namedtuple('DataIO', 'input output')


def io_split(df, input_cols, output_cols):
    """ Take a dataframe and split it into input and output columns.
        Currently just hardcoding split by columns
    """
    return DataIO(df.loc[:, input_cols].values, df.loc[:, output_cols].values)
