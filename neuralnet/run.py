from neuralnet.splitter import Splitter
from neuralnet.sequential import SequentialNeuralNet
import pandas as pd
from preprocessing.utils import printv
import matplotlib.pyplot as plt
from neuralnet.models import make_model
import math
from config import GRAPH_PATH, Column
import os
from neuralnet.grapher import Grapher


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

DEFAULT_MODEL = "BaselineModel"


class NeuralNetRunner(object):
    def __init__(self, args, nnmodel=None, modelfile=None,
                 save_model=True, show_graph=True):
        self.args = args
        self.model_name = args.nnmodel if not nnmodel else nnmodel
        self.model = None
        self.modelfile = None
        if save_model:
            self.modelfile = args.modelfile
        self.show_graph = show_graph

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
        with SequentialNeuralNet(self.model, self.modelfile) as net:
            net.train(tr)
            output, loss = net.predict(ts.input, ts.output)
            predicted = pd.DataFrame(output, columns=self.model.output_cols)
        predicted = self.model.post_process(ts, predicted)
        printv("Predicted data like:\n", predicted[:5])
        if self.show_graph:
            Grapher(self.args.graphfile).graph(ts, predicted)
        return loss

