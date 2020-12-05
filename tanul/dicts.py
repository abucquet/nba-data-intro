# This file contains a myriad of dictionaries used in other files for layer types,
# activation functions, loss functions...


##### TODO: add ability to use custom losses?
# - either as an option
# - either by just modifiying these dicts
# BEST: 
# add a "custom" key that recognizes that teh user wants a custom loss/optim/activation

import torch.nn as nn
import torch.optim as optim

ACTIVATION_DICT = {
	"relu": nn.ReLU(),
	"sigmoid": nn.Sigmoid(),
	"tanh": nn.Tanh(),
	"softmax": nn.Softmax(),  
	"softplus": nn.Softplus(),
	"DEFAULT": nn.ReLU()
}

CRITERION_DICT = {
	"l1": nn.L1Loss(),
	"mse": nn.MSELoss(),
	"cross entropy": nn.CrossEntropyLoss(),
	"binary cross entropy": nn.BCELoss(),
	"nll": nn.NLLLoss(),  
	"poisson nll": nn.PoissonNLLLoss(),
	"kl divergence": nn.KLDivLoss(),
	"DEFAULT": nn.MSELoss()
}


OPTIMIZER_DICT = {
	"adadelta": optim.Adadelta,
	"adagrad": optim.Adagrad,
	"adam": optim.Adam,
	"asgd": optim.ASGD,
	"lbfgs": optim.LBFGS,  
	"rmsprop": optim.RMSprop,
	"sgd": optim.SGD,
	"DEFAULT": optim.SGD
}