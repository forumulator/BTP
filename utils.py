

def ratio_name(na, nb):
    return str(na) + "_" + str(nb)


def get_column_ratio(df, col1, col2):
    return df[col1] / df[col2]


def make_print(verbose):
    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return printv
