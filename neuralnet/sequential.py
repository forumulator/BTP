import keras
from keras.models import Sequential
from keras.layers import Dense
from preprocessing.utils import printv
import time


DEFAULT_SAVE_FILE = "net.json"
INPUT_DIMS = 3
OUTPUT_DIMS = 5


class SequentialNeuralNet(object):
    def __init__(self, model, save=False, save_file=None):
        # The neural net object
        self._net, self._model = Sequential(), model
        self._setup()
        self.save, self.save_file = save, save_file

    def _setup(self):
        printv("Setting up sequential neural net")
        self._model.make_net(self._net)
        return

    def __enter__(self):
        self.compile()
        return self

    def compile(self):
        printv("Compiling network")
        self._net.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, tr_data, batch_size=10, epoch=100):
        """ tr_data is a names tuple netbuilder.DataIO, which contains
            the fields input, and output, which are pandas data frames
            of the input and output columns respectively
        """
        printv("Training netowrk for (batch: %d, epoch: %d)"
               % (batch_size, epoch))
        self._net.fit(tr_data.input.values, tr_data.output.values,
                      batch_size=batch_size, nb_epoch=epoch)
        printv("Done training")

    def predict(self, ts_data):
        printv("Predicting using model")
        pred_output = self._net.predict(ts_data.values)
        printv("Predicted data like: ", pred_output[:5])
        return pred_output

    def save_net(self, out_file):
        if not out_file:
            raise ValueError("Output file can't be null")
        with open(out_file, "w") as out:
            out.write(self._net.to_json())

    @staticmethod
    def rand_net_filename():
        return "net__" + str(time.time()) + ".json"

    def __exit__(self, exc_type, exc_value, traceback):
        if self.save:
            self.save_net(self.save_file if self.save_file \
                          else SequentialNeuralNet.rand_net_filename())
