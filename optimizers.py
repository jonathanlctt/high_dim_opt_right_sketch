import torch
import torch.optim as optim

from utils import jacobian, hessian, conjugate_gradient




class SGD(optim.SGD):

	def __init__(self, params, **optim_params):

		optim.SGD.__init__(self, params, **optim_params)


	def apply_step(self, *losses):
		loss = losses[0]
		loss.backward()
		self.step()



class Adagrad(optim.Adagrad):

	def __init__(self, params, **optim_params):

		optim.Adagrad.__init__(self, params, **optim_params)


	def apply_step(self, *losses):
		loss = losses[0]
		loss.backward()
		self.step()



class Adam(optim.Adam):

	def __init__(self, params, **optim_params):

		optim.Adam.__init__(self, params, **optim_params)


	def apply_step(self, *losses):
		loss = losses[0]
		loss.backward()
		self.step() 



class Newton:

	def __init__(self, params, **optim_params):
		self.params = params
		self.lr = optim_params['lr']
		self.tol = optim_params['tol']
		self.n_cg = optim_params['n_cg']


	def zero_grad(self):
		pass


	def apply_step(self, *args):
		loss_g, loss_h = args[:2]

		for x in self.params:			
			g = jacobian(loss_g, x)
			h = hessian(loss_h, x)

			with torch.no_grad():
				g = g.reshape((-1,1))
				h = h.reshape((g.shape[0], g.shape[0]))
				dx = conjugate_gradient(h, g, n_iterations=self.n_cg, tol=self.tol).reshape(x.shape)
				x.add_(dx, alpha=-self.lr)









