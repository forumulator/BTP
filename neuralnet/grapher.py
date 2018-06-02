from config import GRAPH_PATH
import matplotlib.pyplot as plt
from preprocessing.utils import printv
import os, math


class Grapher:
    """ To graph the values output from the neural net """
    def __init__(self, graphfile):
        self.graphfile = graphfile

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
        save = self.graphfile if self.graphfile\
            else input("Save Graph (Enter filename to save, blank to skip)?: ")
        if save:
            self._save_graph(fig, save)

    def _save_graph(self, fig, filename):
        """ Save fig to filename """
        if not filename.endswith(".png"):
            filename += ".png"
        printv("Saving graph to: " + filename)
        fig.savefig(os.path.join(GRAPH_PATH, filename))

    def _graph_column(self, col_name, ax, actual, predicted):
        ax.plot(actual, color='b')
        ax.plot(predicted, color='g')
        ax.set_title(col_name)
        ax.legend(["Actual values", "Predicted Values"])