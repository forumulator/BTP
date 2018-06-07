from neuralnet.run import NeuralNetRunner
from prettytable import PrettyTable

class ModelComparator(object):
	""" Comparator for different neural network models """
	def __init__(self, model_list):
		self.model_list = model_list

	def _tabulate(self, metrics):
		table = PrettyTable(self.model_list)
		table.add_row(metrics)
		print(table)

	def compare(self, df):
		metrics = []
		try:
			for model in self.model_list:
				metrics.append(
					NeuralNetRunner(None, model, modelfile=None, save_model=False,
									show_graph=False).run(df))
		except ValueError as e:
			print(str(e))
			return
		self._tabulate(metrics)
