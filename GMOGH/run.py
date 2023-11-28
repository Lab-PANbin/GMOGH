import numpy as np
import torch
from torch.optim import Adam
from functions import get_gradient
from pymoo.factory import get_problem
from obj_function import *
import time
from sklearn.linear_model import LinearRegression
#from functions import sre
from functions import binary
from scipy.io import loadmat

#load data
data = loadmat("GMOGH/Urban_F210.mat")
data1 = loadmat('GMOGH/data_example.mat')
data2 = loadmat('GMOGH/end6_groundTruth.mat')
cur_problem = 'zdt1'
run_num = 0

if __name__ == "__main__":
    k = 30
    x = torch.rand((498, k))
    x1 = torch.zeros_like(x)
    
    x.requires_grad = True
    optimizer = Adam([x], lr=5e-3)
    iters = 1000
    start_time = time.time()

    for i in range(iters):
        loss_1, loss_2 = loss_function(x, problem=cur_problem)
        print(loss_1, loss_2)
        pfront = torch.cat([loss_1.unsqueeze(0), loss_2.unsqueeze(0)], dim=0)
        print(pfront)
        pfront = pfront.detach().cpu().numpy()
      
         
        if i%1000 == 0:
            problem = get_problem(cur_problem)
            x_p = problem.pareto_front()[:, 0]
            y_p = problem.pareto_front()[:, 1]
            
        loss_1 = loss_1.float()
        loss_1.sum().backward(retain_graph=True)
        grad_1 = x.grad.detach().clone()
        x.grad.zero_()

        loss_2 = loss_2.float()
        loss_2.requires_grad_(True) 
        loss_2.sum().backward(retain_graph=True)
        grad_2 = x.grad.detach().clone()
        x.grad.zero_()
       
        
        grad_1 = torch.nn.functional.normalize(grad_1, dim=0)
        grad_2 = torch.nn.functional.normalize(grad_2, dim=0)

        optimizer.zero_grad()
        losses = torch.cat([loss_1.unsqueeze(0), loss_2.unsqueeze(0)], dim=0)
        print(losses)
        x.grad = -get_gradient(grad_1, grad_2, x, losses)
        optimizer.step()
        
        x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)
        x1.data = torch.zeros_like(x.data)
        for i in range(k):
            x1.data[:,i] = binary(x.data[:,i].clone())

    print(i, 'time:', time.time()-start_time, loss_1.sum().detach().cpu().numpy(), loss_2.sum().detach().cpu().numpy(),"x.data",x.data,"x1",x1.data)
    #np.savetxt(r'endmember_ur.txt', x1.data[:,1])
    #np.savetxt(r'endmember_ur1.txt', x.data[:,1])

    x_init=x1.data[:,1].numpy()
    x1= np.array(x_init)
    A = np.array(data1["A"][0:210,0:498])
    As = np.dot(A,np.diag(x1))
    x = np.array(As)
    # observation
    y = np.array(data["Y"])
    #得到丰度
    reg_nnls = LinearRegression(positive=True)
    G = reg_nnls.fit(x, y).coef_
    np.savetxt(r'u_gmogh_m.txt', G)

    #print("abundance",G)
    #得到重构图
    y1 = np.dot(x,G.T)
    


