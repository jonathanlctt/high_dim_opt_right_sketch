import torch
from time import time




def timeit(method):
    def timed(*args, **kwargs):
        start = time()
        result = method(*args, **kwargs)
        end = time()
        return result, end-start
    return timed



def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
               


def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)



def conjugate_gradient(A, b, n_iterations=100, tol=1e-5):

    d = b.shape[0]
    x = torch.zeros((d,1))
    r = b.clone()
    p = b.clone()

    for _ in range(n_iterations):
        r2 = (r**2).sum()
        Ap = torch.mm(A, p)
        alpha = r2 / (p*Ap).sum()
        x += alpha * p
        r -= alpha * Ap 
        if torch.norm(r) < tol:
            return x
        beta = (r**2).sum() / r2
        p = r + beta * p

    return x







