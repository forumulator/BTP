

def make_print(verbose):
    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return printv