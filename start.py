import pandas as pd
import argparse
from preprocessing.cleanup import preprocess_data
from preprocessing.basicgraph import basic_graph
from preprocessing import utils
from neuralnet import NeuralNetRunner

DATA1 = "data1.csv"
TO_GRAPH = False


def main(args):
    df = pd.read_csv(DATA1)
    # Basic cleanup
    utils.printv("\n### Preprocessing data ###\n")
    df = preprocess_data(df)
    if args.basic_graph or TO_GRAPH:
        utils.printv("\n### Graphing Data ###\n")
        basic_graph(df)
    # Make an graph the neural net
    utils.printv("\n### Running the network ###\n")
    runner = NeuralNetRunner()
    runner.run(df)



def parseargs():
    parser = argparse.ArgumentParser(description='Cleanup and processing'
                                                 ' of the RO dataset')
    parser.add_argument('-b', '--basic_graphs', dest="basic_graph",
                        default=False, action="store_true",
                        help='Draw basic analytic graphs')
    # parser.add_argument('-b')
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