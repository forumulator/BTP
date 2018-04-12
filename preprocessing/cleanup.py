import pandas as pd
from preprocessing.utils import printv
from config import Column, col_names
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Preprocessor(object):
    def __init__(self, verbose=False, verbose_func=None):
        self.verbose = verbose
        self.verbose_func = verbose_func

    def process(self, df):
        df = self._processor(df)
        if self.verbose:
            if not self.verbose_func:
                printv("Values after passing through %s like: "
                      % self.__class__.__name__)
                printv(df[:5])
            else:
                self.verbose_func(df)
        return df

    def _processor(self, df):
        raise NotImplemented("Not implemented here")


class NanValueFilter(Preprocessor):
    def __init__(self, method='ffill', verbose=False, verbose_func=None):
        super().__init__(verbose)
        self.method = method

    def _processor(self, df):
        df = df.fillna(method=self.method)
        return df


class ColumnFilter(Preprocessor):
    _REM_COLS = [
        Column.ROBW, Column.DATETIME,
        Column.REJ_VOL_ML, Column.BACKWASH_WATER
    ]

    def __init__(self, verbose=False, verbose_func=None):
        super().__init__(verbose, verbose_func)

    def _processor(self, df):
        return self._remove_unused_cols(self._rename_cols(df))

    def _remove_unused_cols(self, df):
        return df.drop(self._REM_COLS, axis=1)

    @staticmethod
    def _rename_cols(df):
        return df.rename(columns=ColumnFilter.name_tr)

    @staticmethod
    def name_tr(name):
        name = name.strip(r"[, ]+")
        if name in col_names:
            new_name = col_names[name]
        else:
            new_name = name.lower().split('(')[0]
        return new_name


class TypeProcessor(Preprocessor):
    _FLOAT_COLS = [
        Column.PERM_FLOWRATE, Column.PERM_MASS,
        Column.PERM_TIME, Column.PERM_TOTAL_FLOW,
        Column.REJ_VOL, Column.MEMBR_FEED_FLOWRATE,
        Column.RECOVERY, Column.REJ_TDS
    ]
    _INT_COLS = [
        Column.TIME, Column.MEMBR_REJ_TDS,
        Column.TANK_TDS, Column.MEMBR_FEED_TDS,
        Column.MEMBR_FEED_PRESSURE
    ]

    def __init__(self, verbose=False, verbose_func=None):
        super().__init__(verbose, verbose_func)

    @staticmethod
    def _fix_rej_tds(df):
        df[Column.REJ_TDS] = df[Column.REJ_TDS].apply(lambda x: str(x).split()[0])

    def _processor(self, df):
        return self._reinterpret_dtypes(df)

    def _reinterpret_dtypes(self, df):
        TypeProcessor._fix_rej_tds(df)
        df[Column.RECOVERY] = pd.to_numeric(df[Column.RECOVERY],
                                            errors='coerce')\
            .fillna(method="ffill")
        df[TypeProcessor._FLOAT_COLS] = df[TypeProcessor._FLOAT_COLS]\
            .astype(float)
        df[TypeProcessor._INT_COLS] = df[TypeProcessor._INT_COLS]\
            .astype(float)
        return df


class Normalizer(Preprocessor):
    def __init__(self, verbose=False, verbose_func=None):
        super().__init__(verbose, verbose_func)

    def _processor(self, df):
        return self._scale_columns(df)

    def _fix_rej_tds(self, df):
        """ Convert PPM to PPT """
        df[Column.REJ_TDS] = df[Column.REJ_TDS]\
            .apply(lambda x: x / 1000 if x > 1 else x)
        return df

    def _scale_columns(self, df):
        df = self._fix_rej_tds(df)
        df = pd.DataFrame(MinMaxScaler().fit_transform(df),
                          columns=df.columns)
        return df


def print_columns(df):
    printv(df.columns)


FILTERS = (
    ColumnFilter(verbose=True), NanValueFilter(),
    TypeProcessor(verbose=True, verbose_func=print_columns),
    NanValueFilter(method='bfill'),
    Normalizer(verbose=True)
)


def preprocess_data(df):
    print = printv
    df = df.iloc[1:323]
    for processor in FILTERS:
        printv("Applying filter: %s"
                     % processor.__class__.__name__)
        df = processor.process(df)
        # print(np.isnan(df.any()), np.isfinite(df.all()))
        df.to_csv("temp.csv")

    printv("Filters applied. Final columns:")
    printv(df.columns)
    printv("\nFinal cleaned up data looks like:\n", df[:3])
    printv()
    return df
