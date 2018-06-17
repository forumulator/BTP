from keras.models import Sequential
from preprocessing.utils import printv
import time, os
from config import MODEL_PATH


DEFAULT_SAVE_FILE = "net.json"
INPUT_DIMS = 3
OUTPUT_DIMS = 5


class SequentialNeuralNet(object):
    def __init__(self, model, save_file=None):
        # The neural net object
        self._net, self._model = Sequential(), model
        self._setup()
        self.save_file = save_file

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
        self._net.fit(tr_data.input, tr_data.output,
                      batch_size=batch_size, epochs=epoch)
        printv("Done training")

    def predict(self, ts_data, ts_output=None):
        printv("Predicting using model, with metric: %s"
                % self._net.metrics_names)
        pred_output = self._net.predict(ts_data)
        loss = None
        if ts_output is not None:
            loss = self._net.evaluate(ts_data, ts_output)
            printv("===============================================")
            printv("The loss (with metric %s) is %s"
                    % (self._net.metrics_names, str(loss)))
            printv("===============================================")
        printv("Predicted data like: ", pred_output[:5])
        return (pred_output, loss)

    def save_net(self, out_file):
        printv("Writing neural net to file: " + out_file)
        if not out_file:
            raise ValueError("Output file can't be null")
        out_file += ".json"
        with open(os.path.join(MODEL_PATH, out_file), "w") as out:
            out.write(self._net.to_json())

    @staticmethod
    def rand_net_filename():
        return "net__" + str(time.time()) + ".json"

    def __exit__(self, exc_type, exc_value, traceback):
        if self.save_file:
            self.save_net(self.save_file)
