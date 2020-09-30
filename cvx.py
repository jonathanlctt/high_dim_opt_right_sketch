import torch
import numpy as np 

import optimizers as optim
from utils import timeit



class CVX:


	def __init__(self, A, y, loss_fn, lambda_):

		self.A = A
		if y.ndim == 1:
			y = y.reshape((-1,1))
		self.y = y
		self.n, self.d = A.shape
		self.p = y.shape[1]
		self.loss_fn = loss_fn 
		self.lambda_ = lambda_
		self.atilde = torch.zeros((self.n, self.p))
		self.btilde = torch.zeros((self.d, self.p))

		self.optimizers = {'sgd': optim.SGD, 'adagrad': optim.Adagrad, 'adam': optim.Adam, 'newton': optim.Newton}


	@timeit
	def _solve(self, optimizer, batches, iteration, x):

		optimizer.zero_grad()
		losses = []
		for _batches in batches:
			batch = _batches[iteration % len(_batches)]
			inputs, targets = self.A[batch], self.y[batch]
			z = torch.matmul(inputs, x) + self.atilde[batch]
			loss = self.loss_fn(z, targets) + 0.5 * self.lambda_ * ((x+self.btilde)**2).sum()
			losses.append(loss)
		optimizer.apply_step(*losses)


	def solve(self, n_iterations=10, mode='evaluation', **optim_params):

		x = 1/np.sqrt(self.d) * torch.randn_like(self.btilde)
		x.requires_grad = True

		optimizer = self.optimizers[optim_params['alg']]([x], **optim_params['params'])
		batches = [torch.split(torch.arange(self.n), batch_size) for batch_size in optim_params['batch_sizes']]

		losses = []
		times = []
		for iteration in range(n_iterations):
			_, time_ = self._solve(optimizer, batches, iteration, x)
			times.append(time_)
			if mode == 'evaluation':
				losses.append(self.loss_fn(torch.matmul(self.A, x.data), self.y) + 0.5*self.lambda_*(x.data**2).sum())
		x.requires_grad = False
		return x, losses, times






















