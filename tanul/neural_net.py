"""
This file contains a scikit learn neural network
"""
# Python imports
from tqdm import tqdm
from typing import List, Dict, Tuple

# various python libraries
from sklearn.model_selection import train_test_split
import numpy as np

# torch imports
import torch
from torch import tensor, randn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# tanul imports
from dicts import ACTIVATION_DICT, CRITERION_DICT, OPTIMIZER_DICT
from pytorch_utils import DatasetFromNumpy

print("ALL IMPORTS SUCCESSFUL")

class NeuralNet(nn.Module):
	def __init__(
		self, 
		input_size: int = None, 
		output_size: int = None, 
		hidden_sizes: List[int] = None, 
		params: Dict = None):
		"""
		In the constructor we instantiate:
		- the linear layers of the Neural Net
		- the optimizer, loss function and the actiavtion functions

		Arguments:
			input_size: int, size of the input
			output_size: int, desired size of the output
			hidden_sizes: List of integers, sizes of the hidden layers
			params: Dict containing customization directions for the Network:
				"input_size": see above
				"output_size": see above
				"hidden_sizes": see above
				"activation": pytorch activation function for the network 
					(Defaults to nn.ReLU)
				"final activation": pytorch activation function for the last layer 
					of the network (Defaults to nn.ReLU or "activation")
				"criterion": pytorch loss criterion to use for training 
					(Defaults to nn.MSELoss)
				"optimizer": pytorch optimizer to use for training 
					(Defaults to optim.SGD)
				"lr": int, learning rate to use
					(Defaults to 10^-4)

			NOTE: one of params or input_size-output_size-hidden_sizes must not be None

		Returns:
			An instance of the NeuralNet class
		"""

		super(NeuralNet, self).__init__()

		# initialize layer sizes
		if params is not None:
			# layer sizes
			self.input_size = params["input_size"]
			self.output_size = params["output_size"]
			self.hidden_sizes = params["hidden_sizes"]

		else: # use the specified inputs, default settings
			self.input_size = input_size
			self.output_size = output_size
			self.hidden_sizes = hidden_sizes
			params = {}
			
		# make hidden layers
		self.hidden_layers = nn.ModuleList()
		# first layer
		self.hidden_layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
		for k in range(len(self.hidden_sizes)-1):
			self.hidden_layers.append(
				nn.Linear(self.hidden_sizes[k], self.hidden_sizes[k+1])
				)
		# last layer
		self.hidden_layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

		# activations
		# TODO: add custom argument options for activations: reduction
		self.activation = self.parse_param(
			"activation", params, ACTIVATION_DICT
		)
		if "final activation" in params:
			self.final_activation = ACTIVATION_DICT[params["final activation"]]
		else:
			self.final_activation = self.activation

		# loss function
		# TODO: ADD REDUCTION ARGUMENT CUSTOMIZATION
		self.criterion = self.parse_param(
			"criterion", params, CRITERION_DICT
		)

		# optimizer and lr
		optimizer = self.parse_param(
			"optimizer", params, OPTIMIZER_DICT
		)
		lr = 1e-4
		if "lr" in params: lr = params["lr"]
		self.optimizer = optimizer(self.parameters(), lr=lr)

	def parse_param(
		self,
		param_name: str, 
		params: Dict, 
		lookup_dict: Dict
		):
		"""
		Grabs the parameter value if it was passed it through the param dict,
		or returns the default parameter value.
		"""
		if (params is not None) and (param_name in params):
			return lookup_dict[params[param_name]]
		else:
			return lookup_dict["DEFAULT"]

	def forward(self, x: tensor):
		"""
		Makes a forward pass in the network by applying layers + activations

		Arguments:
			x: torch.tensor, input to te net

		Returns:
			y_pred: torch.tensor, output of the network
		"""
		for i, mod in enumerate(self.hidden_layers):
			x = mod(x)
			if i == len(self.hidden_sizes):
				x = self.activation(x)
			else:
				x = self.final_activation(x)
		return x

	def train_batch(self, x: tensor, y: tensor):
		"""
		Trains the network using the passed-in batch

		Arguments:
			x: torch.tensor, input to the network
			y: torch.tensor, ground-truth label
		"""
		# forward
		y_pred = self.forward(x)
		
		# loss
		loss = self.criterion(y_pred, y)
		
		# backward
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def score(self, loader: DataLoader) -> float:
		"""
		Scores the network's performance on the passed-in data

		Arguments:
			loader: torch DataLoader, data batches
		Returns:
			loss: float, loss of the network on the (x, y) pair
		"""
		self.eval()
		loss = 0

		with torch.no_grad():
			for x, y in loader:
				# forward
				y_pred = self.forward(x)
			
				# loss
				loss += self.criterion(y_pred, y).numpy()

		self.train()

		return loss

	def fit(
		self, 
		x: np.array,
		y: np.array,
		max_epochs: int = 10,
		batch_size: int = 32,
		train_val_split: float = 0.33,
		verbose: bool = True,
		progress: bool = False,
		):

		"""
		Does the whole training routine for the network.

		Arguments:
			x: np.array, predictors
			y: np.array, ground truths
			max_epochs: int, max number of epochs to run (defaults to 10)
			batch_size: int, number of observations in a batch
			train_val_split: float, fraction of observation to use as validation (defaults to 0.33)
			verbose: bool, whether or not to print validation scores (defaults to True)
			progress: bool, whether or not to show progress bar (defaults to False)
		"""
		all_epochs = range(max_epochs)
		if progress:
			all_epochs = tqdm(range(max_epochs))

		train_loader, val_loader = self.make_data_loaders(
			x, y, batch_size, train_val_split
		)

		for e in all_epochs:
			for x_batch, y_batch in train_loader:
				self.train_batch(x_batch, y_batch)
			
			# score the validation set
			if verbose:
				val_loss = self.score(val_loader)
				print("Epoch {}/{}: Validation Loss {}".format(e, max_epochs, val_loss))


	def make_data_loaders(
		self,
		x: np.array, y: np.array,
		batch_size: int, 
		train_val_split: float
	) -> Tuple[DataLoader]:
		"""
		TODO: write a comment :)
		"""
		x_train, x_val, y_train, y_val = train_test_split(
			x, y, test_size=train_val_split
		)

		# train loader
		train_set = DatasetFromNumpy(x_train, y_train)
		train_loader = DataLoader(
			train_set, batch_size=batch_size, shuffle=True
		)

		# val loader
		val_set = DatasetFromNumpy(x_val, y_val)
		val_loader = DataLoader(
			val_set, batch_size=batch_size, shuffle=False
		)

		return train_loader, val_loader

#### TESTING CODE
# 1. Test basic network, all defaults
N, in_size, hid_sizes, out_size = 64, 100, [100], 10
x = np.random.random(size=(N, in_size))
y = np.random.random(size=(N, out_size))

net = NeuralNet(input_size=in_size, output_size=out_size, hidden_sizes=hid_sizes)
net.fit(x, y)
print("Test: Basic Architecture -- PASSED")

# 2. Test param dict
params = {
	"input_size": in_size, 
	"output_size": out_size, 
	"hidden_sizes": hid_sizes
}
net = NeuralNet(params=params)
net.fit(x, y)
print("Test: Param Dict -- PASSED")

# 3. Test more layers
params = {
	"input_size": in_size, 
	"output_size": out_size, 
	"hidden_sizes": [1000, 200, 5],
	"activation": "relu"
}
net = NeuralNet(params=params)
net.fit(x, y)
print("Test: Three layer MLP -- PASSED")


















