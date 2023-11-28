import math
import torch
import numpy as np


def solve_min_norm_2_loss(grad_1, grad_2):
    v1v1 = torch.sum(grad_1*grad_1, dim=1)
    v2v2 = torch.sum(grad_2*grad_2, dim=1)
    v1v2 = torch.sum(grad_1*grad_2, dim=1)
    gamma = torch.zeros_like(v1v1)
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    gamma[v1v2>=v1v1] = 0.999
    gamma[v1v2>=v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    g_w = gamma.repeat(1, grad_1.shape[1])*grad_1 + (1.-gamma.repeat(1, grad_2.shape[1]))*grad_2

    return g_w

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_functional_rbf(losses):
    n = losses.shape[0]
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=0).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / h) 
    return kernel_matrix

def get_gradient(grad_1, grad_2, inputs, losses):
    n = inputs.size(0)
   
    g_w = solve_min_norm_2_loss(grad_1, grad_2)
    
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_functional_rbf(losses)

    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]
    gradient = (g_w - kernel_grad) / n

    return gradient

def binary(data):
    beta = 0.7
    for i in range(len(data)):
        if data[i]<beta:
            data[i]=0
        if data[i]>beta:
            data[i]=1
    return data

def sre(ypred, y):
    sre_final = []
    for i in range(y.shape[1]):
        numerator = np.square(np.mean(y[:, i]))
        denominator = (np.linalg.norm(y[:, i] - ypred[:, i])) /\
                      (y.shape[0])
        sre_final.append(numerator/denominator)

    return 10 * np.log10(np.mean(sre_final))

