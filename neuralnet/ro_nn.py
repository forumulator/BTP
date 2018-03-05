import pandas as pd
import numpy
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Importing Keras.")
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout

numpy.random.seed(7)

# The netural network class for the primary
# entity detection. Currently the features used are:
# 1. Vectors on the left and Right
# 2. Frequency of the entity
# class RoNeuralNet:
#   def __init__(self):
#     self.nets = None
#     self.tr_data, self.tr_out = {}, {}
# 	model = Sequential()
# 	# input_dim = 2* config.word2vec_dim * gen_ent_vectors.radius + 2
# 	input_dim = 82
# 	model.add(Dense(12, input_dim = input_dim, activation='relu'))
# 	# model.add(Dropout(0.3))
# 	model.add(Dense(8, activation='relu'))
# 	# model.add(Dropout(0.3))
# 	# model.add(Dense(50, activation='relu'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	self.nets[catg] = model
# 	self.tr_data[catg] = []
# 	self.tr_out[catg] = []

#   def compile(self):
#     for catg in self.nets:
#       self.nets[catg].compile(loss='binary_crossentropy',
#         optimizer='adam', metrics=['accuracy'])

#   # Add a new data point for training
#   def add_training_data(self, catg, ent_veclist_back,
#       ent_veclist_forw, freq, n_pos, relevant):
#     # Convert it all to one vector
#     feature_vec = None
#     for vec_list in (ent_veclist_back, ent_veclist_forw):
#       for ent_vec in vec_list:
#         if feature_vec is None:
#           feature_vec = ent_vec
#         else:
#           feature_vec = numpy.append(feature_vec, ent_vec)
#     feature_vec = numpy.append(feature_vec, [freq, n_pos])
#     self.tr_data[catg].append(feature_vec)
#     self.tr_out[catg].append(int(relevant))

#   def save_to_file(self, json_file, weights_file):
#     for catg in self.nets:
#       with open("{0}.{1}".format(json_file, catg), "w") as json_f:
#         json_f.write(self.nets[catg].to_json())
#       self.nets[catg].save_weights("{0}.{1}".format(weights_file, catg))

#   def load_from_file(self, json_file, weights_file):
#     for catg in self.nets:
#       with open("{0}.{1}".format(json_file, catg), 'r') as json_f:
#         self.nets[catg] = model_from_json(json_f.read())
#       # load weights into new model
#       self.nets[catg].load_weights("{0}.{1}".format(weights_file, catg))

#   # Train the network on the so far collected input data
#   def train_network(self, epochs = 30, batch_size = 20):
#     for catg in self.nets:
#       print("Training net for category: " + catg)
#       print("Size of " + catg + ": " + str(len(self.tr_data[catg])))
#       self.tr_data[catg] = numpy.array(self.tr_data[catg])
#       self.tr_out[catg] = numpy.array(self.tr_out[catg])
#       self.nets[catg].fit(self.tr_data[catg], self.tr_out[catg],
#         epochs=epochs, batch_size=batch_size,  verbose=2)
#       scores = self.nets[catg].evaluate(self.tr_data[catg], self.tr_out[catg])
#       print("\n%s: %.2f%%" % (self.nets[catg].metrics_names[1], scores[1]*100))

#   def predict(self, catg, ent_veclist_back, ent_veclist_forw, freq, n_pos):
#     feature_vec = None
#     for vec_list in (ent_veclist_back, ent_veclist_forw):
#       for ent_vec in vec_list:
#         if feature_vec is None:
#           feature_vec = ent_vec
#         else:
#           feature_vec = numpy.append(feature_vec, ent_vec)
#     feature_vec = numpy.append(feature_vec, [freq, n_pos])
#     predictions = self.nets[catg].predict(numpy.array([feature_vec]))
#     # round predictions
#     rounded = [int(round(x[0])) for x in predictions]
#     return bool(rounded[0])

def load_ro_data(file_name):
	df = pd.read_csv(file_name, header = None, skiprows = 2)
	df = df[:924]
	rm_cols = ['Date & Time', 'Backwash water', 'Reject water', 'RO/BW']
	# for col_name in rm_cols:
	df.drop(df.columns[[0, 8, 12, 21]], axis = 1, inplace = True)
	return df.as_matrix()

def baseline_model():
	model = Sequential()
	input_dim = 16
	model.add(Dense(12, input_dim = input_dim, 
		kernel_initializer = "normal", activation='relu'))
	# model.add(Dropout(0.3))
	# model.add(Dense(8, activation='relu'))
	model.add(Dense(1, kernel_initializer = "normal"))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

RO_DATA_FILE = "ro_data.csv"
def main():
	inp = load_ro_data(RO_DATA_FILE).astype(numpy.float32)
	numpy.random.shuffle(inp)
	# Seperate data into training and test using
	# 800 and 124 respectively
	train, test = inp[:800, :], inp[800:, :]
	
	print("Now training neural net...")
	out_cols = [3, 5]
	tr_out = [train[:, i] for i in out_cols]
	tr_in = numpy.delete(train, out_cols, axis = 1)
	# evaluate model with standardized dataset
	estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=20, batch_size=10, verbose=0)
	kfold = KFold(n_splits=10, random_state=7)
	results = cross_val_score(estimator, tr_in, tr_out[0], cv=kfold)
	fig = plt.figure()
	p1 = plt.plot(results)
	plt.show()
	print(results)
	print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

	# model.fit(tr_in, tr_out[0],
	# 	epochs=10, batch_size=10, verbose=2)
	# scores = model.evaluate(tr_in, tr_out[0])
	# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# print("Done training. Now testing...")
	# ts_out = [test[:, i] for i in out_cols]
	# ts_in = numpy.delete(test, out_cols, axis = 1)
	# print(ts_in, ts_out[0])
	# scores = model.evaluate(ts_in, ts_out[0])
	# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	# print(ts_in.shape, ts_in[0].shape)
	# print(model.predict(numpy.array([ts_in[0]])))




if __name__ == "__main__":
	main()