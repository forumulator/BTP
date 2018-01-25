import pandas as pd
import utils
from ro_data import Column
from sklearn.preprocessing import MinMaxScaler


def unnamed(i):
    return 'Unnamed: ' + str(i)


col_names = {
    'Permeate properties': 'perm_mass',
    unnamed(3): 'perm_time',
    unnamed(4): 'perm_cond',
    unnamed(5): 'perm_totalflow',
    unnamed(6): 'perm_flowrate',
    'Total Rej': 'reject_total',
    # 'Reject water': '',
    unnamed(11): 'recovery',
    'Reject from membarne': 'reject_tds',
    unnamed(14): 'reject_flowrate_ml',
    unnamed(15): 'reject_flowrate_l',
    'Tank': 'tank_tds',
    'Feed to membrane': 'membr_tds',
    unnamed(18): 'membr_flow',
    unnamed(19): 'membr_pressure',
    'Feed Inlet to recycle tank': 'inlet_flow',
    unnamed(21): 'inlet_tds',
}


def rename_cols(name):
    name = name.strip()
    if name in col_names:
        return col_names[name]
    return name.lower().split('(')[0]


float_cols = [Column.PERM_FLOWRATE, Column.PERM_MASS,
              Column.PERM_TIME, Column.PERM_TOTAL_FLOW,
              Column.REJ_TOTAL, Column.MEMBR_FEED_FLOW]
int_cols = [Column.TIME, Column.MEMBR_REJ_FLOWRATE, Column.MEMBR_REJ_TDS,
            Column.TANK_TDS, Column.MEMBR_FEED_TDS,
            Column.MEMBR_FEED_PRESSURE]


def reinterpret_dtypes(df):
    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].astype(float)


def scale_columns(df):
    df[Column.TANK_TDS] = \
        pd.DataFrame(MinMaxScaler().fit_transform(df),
                     columns=[Column.TANK_TDS])

def basic_cleanup(df):
    df = df.iloc[1:323]
    df.fillna(method="ffill")
    df.rename(columns=rename_cols, inplace=True)
    # Data types
    reinterpret_dtypes(df)
    #Scaling
    # scale_columns(sd)
    utils.printv(df.columns)
    utils.printv(df[:20])
    return df