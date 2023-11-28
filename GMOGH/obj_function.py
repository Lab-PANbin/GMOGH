import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression

#load data
data = loadmat("GMOGH/Urban_F210.mat")
data1 = loadmat('GMOGH/data_example.mat')

k=30
library= data1["A"][0:210,0:498] #224*498
observation= data['Y']/1.0 #224*47500

library = torch.from_numpy(library).float()
observation = torch.from_numpy(observation).float()

pinverse =  torch.pinverse(library)
parameter = torch.mm(pinverse,observation)

def loss_function(x, problem='zdt1'):
    f = 0
    g = 0
    if problem == 'zdt1':
        
        for i in range(30):
            diag=torch.diag_embed(x[:,i])#498*498
            mm = torch.mm(library,diag)
            As=torch.nan_to_num(mm)   
            # #设置目标函数
            value=torch.nan_to_num(observation-torch.mm(As,parameter))
            v = torch.norm(value)
            f = f+v
            i = i+1
        f = f/k
    
        num=0
        numj = 0
        for i in range(30):
            for j in range(0, 498):
                if x[:, i][j] > 0:
                    numj += 1
            num = num+ numj   
        num = np.array(num)
        num = torch.from_numpy(num).float()
        g = torch.abs(num-6)/k
    return f, g


