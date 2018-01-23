import pandas as pd
import matplotlib.pyplot as plt
import argparse
from cleanup import basic_cleanup
from basicgraph import basic_graph
import utils

DATA1 = "data1.csv"


def main(args):
    df = pd.read_csv(DATA1)
    # Basic cleanup
    df = basic_cleanup(df)
    if args.basic_graph:
        basic_graph(df)


def parseargs():
    parser = argparse.ArgumentParser(description='Cleanup and processing'
                                                 ' of the RO dataset')
    parser.add_argument('-b', '--basic_graphs', dest="basic_graph",
                        default=False, action="store_true",
                        help='Draw basic analytic graphs')
    parser.add_argument('-v', '--verbose', dest="verbose",
                        default=False, action="store_true",
                        help='Draw basic analytic graphs')
    parser.add_argument('-l', '--lim', dest='limit', type=int, default=-1,
                        help='Limit on the number of directories to process'
                             ' (default is all directories)')
    return parser.parse_args()


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    args = parseargs()
    # Make printv function
    utils.printv = utils.make_print(args.verbose)
    main(args)