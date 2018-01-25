from ro_data import Column
import matplotlib.pyplot as plt


class BasicGrapher(object):
    def __init__(self, df):
        """ Input is a cleaned up dataframe"""
        self.df = df

    def draw_time_graphs(self):
        # self.df[[Column.PERM_MASS, Column.PERM_FLOWRATE]]\
        #     .plot(x=Column.TIME)
        # print( self.df[:20].set_index('time'))
        self.df.set_index(Column.TIME)[
                [Column.PERM_FLOWRATE,
                 Column.PERM_MASS,
                 Column.TANK_TDS]].plot()
        # fig, ax = plt.subplots()
        # self.df.plot(x=Column.TIME, y=Column.PERM_FLOWRATE, color="red", ax=ax)
        # self.df.plot(x=Column.TIME, y=Column.PERM_MASS, color="blue", ax=ax)
        # self.df.plot(x="time", y=[Column.PERM_FLOWRATE,
        #                           Column.PERM_MASS],
        #              style='o')


def basic_graph(df, args=None):
    grapher = BasicGrapher(df)
    grapher.draw_time_graphs()
    plt.show()