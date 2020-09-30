import torch
import numpy as np

from cvx import CVX
from srht import srht
from utils import timeit



class sketchCVX(CVX):


    def __init__(self, A, y, loss_fn, lambda_):

        CVX.__init__(self, A, y, loss_fn, lambda_)
    

    @timeit
    def sketch(self, **params):

        adaptive, sketch, m, power = params['adaptive'], params['sketch'], params['sketch_size'], params['power']

        if adaptive:
            if sketch == 'srht':
                S = srht(self.A, m)[0].T
            elif sketch == 'gaussian':
                S = torch.matmul(self.A.T, 1./np.sqrt(m)*torch.randn(self.n, m))
            _norm = torch.norm(self.A)
            for _ in range(power-1):
                S = torch.matmul(self.A.T, torch.matmul(self.A, S)) / _norm**2
        else:
            if sketch == 'srht':
                S = srht(self.A.T, m)[1].T
            elif sketch == 'gaussian':
                S = 1./np.sqrt(m)*torch.randn(self.d, m)
                
        U, _, V = torch.svd(S, some=True)
        Swhite = torch.matmul(U, V.T)
        ASdagger = torch.matmul(self.A, Swhite)
        self.A_ = self.A.clone()
        self.A = ASdagger
        self.Swhite = Swhite


    @timeit
    def reconstruct_solution(self, alpha):
    	alpha.requires_grad = False
    	z = torch.matmul(self.A, alpha) + self.atilde
    	z.requires_grad = True
    	self.loss_fn(z, self.y).backward()
    	return -torch.matmul(self.A_.T, z.grad) / self.lambda_


    def iterative_method(self, n_repetitions=1, n_iterations=10, **optim_params):
        xtilde = torch.zeros(self.A_.shape[1], self.p)
        times = []
        losses = []
        for repetition in range(n_repetitions):
            self.atilde = torch.matmul(self.A_, xtilde)
            self.btilde = torch.matmul(self.Swhite.T, xtilde)
            alpha, losses_, times_ = self.solve(n_iterations, **optim_params)
            xtilde, time_ = self.reconstruct_solution(alpha)
            times.append(times_+[time_]); losses.append(losses_)
        xtilde.requires_grad = False
        return xtilde, losses, times







