from ro_data import Column
import matplotlib.pyplot as plt
import utils


class BasicGrapher(object):
    _PERM_COLUMNS = [
        Column.PERM_FLOWRATE,
        Column.PERM_MASS,
        Column.PERM_COND
    ]
    _INLET_COLUMNS = [
        Column.INLET_FLOWRATE, Column.INLET_TDS
    ]
    _REJECT_COLUMNS = [
        Column.MEMBR_REJ_TDS, Column.MEMBR_REJ_FLOWRATE,
        Column.MEMBR_FEED_PRESSURE
    ]

    def __init__(self, df):
        """ Input is a cleaned up dataframe"""
        self.df = df.copy(deep=True)

    def draw_time_graphs(self):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.title('Normalized values of data vs. time')
        subplots = [
            (BasicGrapher._PERM_COLUMNS, "Permeate Output"),
            (BasicGrapher._INLET_COLUMNS, "Inlet Parameters"),
            (BasicGrapher._REJECT_COLUMNS, "Reject Parameters"),
        ]
        for i, ax in enumerate(ax.reshape(-1)):
            if i == len(subplots): break
            self.df.set_index(Column.TIME)[subplots[i][0]]\
                .plot(ax=ax)
            ax.set_title(subplots[i][1])

    def draw_ratio_graphs(self):
        self.df['io_ratio'] = self.df[Column.MEMBR_REJ_FLOWRATE] \
                              / self.df[Column.INLET_FLOWRATE]
        fig, ax = plt.subplots()
        plt.title("Reject/Inlet Ratio vs. time")
        self.df.set_index(Column.TIME)['io_ratio'].plot(ax=ax)
        print(self.df["io_ratio"][:5])

    def plot(self, block=True):
        utils.printv("Plotting graphs...")
        plt.show(block)


def basic_graph(df, args=None):
    grapher = BasicGrapher(df)
    grapher.draw_time_graphs()
    grapher.draw_ratio_graphs()
    grapher.plot()