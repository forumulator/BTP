from neuralnet.splitter import io_split, Splitter
from neuralnet.sequential import SequentialNeuralNet
import pandas as pd
from preprocessing.utils import printv
import matplotlib.pyplot as plt
from preprocessing.ro_data import Column


INPUT_COLS = [
    Column.TIME, Column.INLET_FLOWRATE,
    Column.INLET_TDS
]
OUTPUT_COLS = [
    Column.PERM_FLOWRATE,
    Column.PERM_COND,
    Column.MEMBR_REJ_FLOWRATE,
    Column.MEMBR_REJ_TDS,
    Column.TANK_TDS
]
TS_SIZE = 0.15


class NeuralNetRunner(object):
    def __init__(self, args=None, save=False, filename=None):
        self.args = args
        self.save, self.filename = save, filename

    def _format_data(self, df):
        """ Needs already cleaned up data """
        # Split into training and test data.
        tr, ts = Splitter(TS_SIZE).split(df)
        # Split columns into input and output
        tr, ts = io_split(tr, INPUT_COLS, OUTPUT_COLS) \
            , io_split(ts, INPUT_COLS, OUTPUT_COLS)
        return tr, ts

    def run(self, df):
        printv("Splitting data into train and test")
        tr, ts = self._format_data(df)
        printv("Data split with element count: (train, test) = (%d, %d)"
               % (len(tr.input), len(ts.input)))
        with SequentialNeuralNet() as net:
            net.train(tr)
            predicted = pd.DataFrame(net.predict(ts.input),
                                     columns=OUTPUT_COLS)
        printv("Predicted data like:\n", predicted[:5])
        self.graph(ts, predicted)

    def graph(self, ts_data, predicted):
        """ Plot the predicted data vs. original data """
        plt.title("Actual vs. predicted values for neural net")
        fig, ax = plt.subplots(nrows=3, ncols=2)
        for i, ax in enumerate(ax.reshape(-1)):
            if i == len(predicted.columns): break
            col_name = predicted.columns[i]
            self._graph_column(col_name, ax, ts_data.output[:, i],
                               predicted.loc[:, col_name])
        plt.show()

    def _graph_column(self, col_name, ax, actual, predicted):
        # printv(ts_data.output)
        # printv(predicted)
        # ax.plot(predicted)
        ax.plot(actual, color='b')
        ax.plot(predicted, color='g')
        ax.set_title(col_name)
        ax.legend(["Actual values", "Predicted values"])
