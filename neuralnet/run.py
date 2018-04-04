from neuralnet.splitter import Splitter
from neuralnet.sequential import SequentialNeuralNet
import pandas as pd
from preprocessing.utils import printv
import matplotlib.pyplot as plt
from neuralnet.models import make_model
import math
from config import GRAPH_PATH, Column
import os


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

DEFAULT_MODEL = "Baseline"


class NeuralNetRunner(object):
    def __init__(self, args, save=False, filename=None):
        self.args = args
        self.model_name, self.model = args.nnmodel, None
        self.save, self.filename = save, filename

    def _format_data(self, df):
        """ Needs already cleaned up data """
        printv("Splitting data into train and test")
        # Split into training and test data.
        tr, ts = Splitter(TS_SIZE).split(df)
        # Split columns into input and output
        tr, ts = self.model.io_split(tr), self.model.io_split(ts)
        printv("Data split with element count: (train, test) = (%d, %d)"
               % (len(tr.input), len(ts.input)))
        return tr, ts

    def run(self, df):
        # Make the neural net model
        if not self.model_name:
            self.model_name = DEFAULT_MODEL
        self.model = make_model(self.model_name)
        # Split data
        tr, ts = self._format_data(df)
        with SequentialNeuralNet(self.model, self.args.modelfile) as net:
            net.train(tr)
            predicted = pd.DataFrame(net.predict(ts.input),
                                     columns=self.model.output_cols)
        printv("Predicted data like:\n", predicted[:5])
        self.graph(ts, predicted)

    def graph(self, ts_data, predicted):
        """ Plot the predicted data vs. original data """
        printv("Graphing predicted values")
        fig, ax = plt.subplots(nrows=int(math.ceil(len(predicted.columns) / 2)),
                               ncols=2)
        fig.suptitle("Actual vs. predicted values for neural net")
        for i, ax in enumerate(ax.reshape(-1)):
            if i == len(predicted.columns): break
            col_name = predicted.columns[i]
            self._graph_column(col_name, ax, ts_data.output[:, i],
                               predicted.loc[:, col_name])
        plt.show()
        # Prompt to save
        save = self.args.graphfile if self.args.graphfile\
            else input("Save Graph (Enter filename to, blank to skip)?:")
        if save:
            self._save_graph(fig, self.args.graphfile)

    def _save_graph(self, fig, filename):
        printv("Saving graph to: " + filename)
        fig.savefig(os.path.join(GRAPH_PATH, self.args.graphfile))

    def _graph_column(self, col_name, ax, actual, predicted):
        ax.plot(actual, color='b')
        ax.plot(predicted, color='g')
        ax.set_title(col_name)
        ax.legend(["Actual values", "Predicted Values"])
