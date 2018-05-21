import pandas as pd
import argparse
from preprocessing.cleanup import preprocess_data
from preprocessing.basicgraph import basic_graph
from preprocessing import utils
from neuralnet import NeuralNetRunner

DATA1 = "data1.csv"
TO_GRAPH = False
NOMODEL = "nomodel"


def main(args):
    df = pd.read_csv(DATA1)
    # Basic cleanup
    utils.printv("\n### Preprocessing data ###\n")
    df = preprocess_data(df)
    if args.basic_graph or TO_GRAPH:
        utils.printv("\n### Graphing Data ###\n")
        basic_graph(df)
    if args.nnmodel != NOMODEL:
        # Make an graph the neural net
        utils.printv("\n### Running the network ###\n")
        runner = NeuralNetRunner(args)
        runner.run(df)


def parseargs():
    parser = argparse.ArgumentParser(
        description='Cleanup, processing,'
                    '  training and graphing the RO dataset')
    parser.add_argument('-b', '--basic_graphs', dest="basic_graph",
                        default=False, action="store_true",
                        help='draw basic analytic graphs')
    # parser.add_argument('-b')
    parser.add_argument('-v', '--verbose', dest="verbose",
                        default=False, action="store_true",
                        help='verbose mode')
    parser.add_argument('-d', '--debug', dest="debug",
                        default=False, action="store_true",
                        help='debug mode')
    parser.add_argument('-l', '--lim', dest='limit', type=int, default=-1,
                        help='limit on the number of directories to process'
                             ' (default is all directories)')
    parser.add_argument(
        '--model', dest='nnmodel', type=str, default="baselinemodel",
        help='neural network model to train on (Default is `baseline\', '
             '`nomodel\' to skip neural nets)')
    parser.add_argument('--save-model', dest='modelfile', type=str, default=None,
                        help='Save model to file MODELFILE')
    parser.add_argument('--save-graph', dest='graphfile', type=str, default=None,
                        help='Save graph to file GRAPHFILE')
    return parser.parse_args()


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    args = parseargs()
    # Make printv function
    utils.printv = utils.make_print(args.verbose)
    utils.debug = args.debug
    main(args)