from config import Column
from sklearn.model_selection import train_test_split
from preprocessing.utils import printv


class Splitter(object):
    def __init__(self, ts_size, shuffle=True):
        self.ts_size = ts_size
        self.shuffle = shuffle

    def split(self, df):
        """ Return a two tuple of tr, ts data """
        printv("Shuffle: " + str(self.shuffle))
        tr, ts = train_test_split(df, test_size=self.ts_size, 
        						  shuffle=self.shuffle)
        return tr, ts


def train_neural_net(df, args=None):
    pass

