import os


MODEL_PATH = "models"
GRAPH_PATH = os.path.join("graphs", "netresults")
DATAFILE = "data2.csv"


def unnamed(i):
    return 'Unnamed: ' + str(i)


class Column(object):
    DATETIME = "date & time"
    TIME = "time"
    ROBW = "ro/bw"
    # Permeate params
    PERM_MASS = "perm_mass"
    PERM_TIME = "perm_time"
    PERM_COND = "perm_cond"
    PERM_TOTAL_FLOW = "perm_totalflow"
    PERM_FLOWRATE = "perm_flowrate"
    # Backwash params
    BACKWASH_WATER = "backwash water"
    BACKWASH = "backwash"
    # Reject water params
    REJ_VOL = "reject_vol"
    REJ_VOL_ML = "reject_vol_ml"
    REJ_TDS = 'reject_tds'
    RECOVERY = "recovery"
    # Membrane Reject params
    MEMBR_REJ_TDS = "membr_reject_tds"
    MEMBR_REJ_FLOWRATE_ML = "membr_reject_flowrate_ml"
    MEMBR_REJ_FLOWRATE = "membr_reject_flowrate_l"
    TANK_TDS = "tank_tds"
    # Membrane feed params
    MEMBR_FEED_TDS = "membr_feed_tds"
    MEMBR_FEED_FLOWRATE = "membr_feed_flow"
    MEMBR_FEED_PRESSURE = "membr_feed_pressure"
    # Inlet flow params
    INLET_FLOWRATE = "inlet_flow"
    INLET_TDS = "inlet_tds"


col_names = {
    'Permeate properties': 'perm_mass',
    unnamed(3): 'perm_time',
    unnamed(4): 'perm_cond',
    unnamed(5): 'perm_totalflow',
    unnamed(6): 'perm_flowrate',
    'Total Rej': 'reject_vol',
    'Reject water': 'reject_vol_ml',
    'Reject water.1': 'reject_tds',
    unnamed(11): 'recovery',
    'Reject from membarne': 'membr_reject_tds',
    unnamed(14): 'membr_reject_flowrate_l',
    # unnamed(15): 'membr_reject_flowrate_l',
    'Tank': 'tank_tds',
    'Feed to membrane': 'membr_feed_tds',
    unnamed(17): 'membr_feed_flow',
    unnamed(18): 'membr_feed_pressure',
    'Feed Inlet to recycle tank': 'inlet_flow',
    unnamed(20): 'inlet_tds',
}

# Index(['date & time', 'time',
# 'perm_mass', 'perm_time', 'perm_cond', 'perm_totalflow', 'perm_flowrate',
# 'backwash water', 'backwash, ',
# 'reject_total', 'reject water', 'recovery', 'reject water.1', 'reject_tds',
# 'reject_flowrate_ml', 'reject_flowrate_l', 'tank_tds', 'membr_tds', 'membr_flow', '
# membr_pressure', 'inlet_flow', 'inlet_tds', 'ro/bw'], dtype='object')