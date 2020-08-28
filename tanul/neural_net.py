"""
This file contains a scikit learn neural network
"""
# Python imports
from tqdm import tqdm
from typing import List, Dict, Tuple

# modeling imports
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim

print("ALL IMPORTS SUCCESSFUL")


class NeuralNet(nn.Module):
	def __init__(
		self, 
		input_size: int = None, 
		output_size: int = None, 
		hidden_size: int = None, 
		params: Dict = None):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""

		super(NeuralNet, self).__init__()

		if params is not None:
			self.input_size = params["input_size"]
			self.output_size = params["output_size"]
			self.hidden_size = params["hidden_size"]
		else:
			self.input_size = input_size
			self.output_size = output_size
			self.hidden_size = hidden_size

		print(self.input_size, params, type(params))

		self.linear1 = nn.Linear(self.input_size, self.hidden_size)
		self.linear2 = nn.Linear(self.hidden_size, self.output_size)

		self.criterion = nn.MSELoss(reduction='sum')
		self.optimizer = optim.SGD(self.parameters(), lr=1e-4)

	def forward(self, x: torch.tensor):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

	def train_batch(self, x: torch.tensor, y: torch.tensor):
		y_pred = self.forward(x)
		loss = self.criterion(y_pred, y)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def train(
		self, 
		x_train: torch.tensor,
		y_train: torch.tensor,
		x_val: torch.tensor = None,
		y_val: torch.tensor = None,
		max_epochs: int = 10
		):

		for e in tqdm(range(max_epochs)):
			self.train_batch(x_train, y_train)



#### TESTING CODE
# 1. Test basic network, all defaults
N, in_size, hid_size, out_size = 64, 1000, 100, 10
x = torch.randn(N, in_size)
y = torch.randn(N, out_size)

net = NeuralNet(input_size=in_size, output_size=out_size, hidden_size=hid_size)
net.train(x, y)
print("Test: Basic Architecture -- PASSED")

# 2. Test param dict
params = {
	"input_size": in_size, 
	"output_size": out_size, 
	"hidden_size": hid_size
}
net = NeuralNet(params=params)
net.train(x, y)
print("Test: Param Dict -- PASSED")


















