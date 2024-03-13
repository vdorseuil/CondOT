from mydataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
import numpy as np
import torch

def get_dataset(d = 2, r = 100, N = 500):
    # simulate the gaussian
    X = torch.tensor(stats.norm.rvs(loc=0, scale=1, size=(N, r, d)), dtype=torch.float32)
    locs = torch.randint(-5, 5, (N,), dtype=torch.float32).repeat_interleave(r).view(N, r) #+ torch.tensor(stats.norm.rvs(loc=0, scale=1, size=(N, r, d)), dtype=torch.float32)

    scales = (0.1 + 5*torch.rand(N, dtype=torch.float32)).repeat_interleave(r).view(N, r)
    #scales = torch.ones(N, r)

    C = torch.stack((locs, scales), dim = 2)

    locs2D = locs.unsqueeze(-1).expand(-1, -1, d)
    scales2D = scales.unsqueeze(-1).expand(-1, -1, d)

    Y = torch.tensor(stats.norm.rvs(loc=locs2D, scale=scales2D, size=(N, r, d)), dtype=torch.float32)

    dataset = MyDataset(X, C, Y)
    # batch_size = batch_size
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # return dataloader, locs, scales
    return dataset

def generate_gaussian_dataset(d = 2, r = 100, N = 500) :
    X = torch.tensor(stats.norm.rvs(loc=0, scale=1, size=(N, r, d)), dtype=torch.float32) 
    
    locs = torch.randint(-3, 4, (N,), dtype=torch.float32).repeat_interleave(r).view(N, r) 
    locs = locs #bruit gaussien
    scales = torch.abs(torch.tensor(stats.norm.rvs(loc=0.9, scale=1, size=(N,)), dtype=torch.float32)).repeat_interleave(r).view(N, r) + 0.1

    locs2D = locs.unsqueeze(-1).expand(-1, -1, d)
    scales2D = scales.unsqueeze(-1).expand(-1, -1, d)

    Y = torch.tensor(stats.norm.rvs(loc=locs2D, scale=scales2D, size=(N, r, d)), dtype=torch.float32)
    Y = torch.abs(Y)

    C = torch.stack((locs, scales), dim = 2)

    dataset = MyDataset(X, C, Y)

    return(dataset)

def generate_dataset(d = 2, r = 100, N = 500) :
    mean_1 = torch.tensor([0,0], dtype=torch.float32).repeat(N,1)
    mean_2 = torch.tensor([2,0], dtype=torch.float32).repeat(N,1)
    cov_1 = [0.2*torch.tensor(np.eye(d)) for i in range(N)]
    cov_2 = [0.2*torch.tensor(np.eye(d)) for i in range(N)]
    X = [torch.concat([torch.tensor(stats.multivariate_normal.rvs(mean=mean_1[i], cov=cov_1[i], size = r)),torch.tensor(stats.multivariate_normal.rvs(mean=mean_2[i], cov=cov_2[i], size = r))]) for i in range(N)]
    X = torch.stack(X)

    ### init facile
    mean_1 = torch.tensor([4,2], dtype=torch.float32).repeat(N,1)
    mean_2 = torch.tensor([2,4], dtype=torch.float32).repeat(N,1)

    cov_1 = [0.2*torch.tensor(np.eye(d)) for i in range(N)]
    cov_2 = [0.2*torch.tensor(np.eye(d)) for i in range(N)]
    
    C = torch.cat([mean_1], dim=1) 


    ### complex init

    # mean_1 = torch.randint(3, 4, (N,2)) #moyenne de la première des deux distributions
    # mean_2 = mean_1.clone() #moyenne de la deuxième des deux distributions
    # mean_2 = torch.randint(2, 3, (N,2))
    cov_1 = torch.tensor(stats.uniform.rvs(0.2, 0.8, size=(N,2,2)))
    cov_1 = [cov_1[0]@cov_1[0].T + 0.1 * torch.tensor(np.eye(d)) for i in range(N)]
    cov_1 = [0 * cov_1[0]@cov_1[0].T + 0.1 * torch.tensor(np.eye(d)) for i in range(N)]
    #cov_2 = torch.tensor(stats.uniform.rvs(-0.8, 0.8, size=(N,2,2)))
    #cov_2 = [cov_2[i]@cov_2[i].T for i in range(N)]
    cov_2 = list(cov_1)

    # cov_1_flatten = cov_1.view(N,4)
    # cov_2_flatten = cov_2.view(N,4)
    C = torch.cat([mean_1, torch.stack(cov_1).view(N,-1), mean_2, torch.stack(cov_2).view(N,-1)], dim=1)
    
    Y = [torch.concat([torch.tensor(stats.multivariate_normal.rvs(mean=mean_1[i], cov=cov_1[i], size = r)),torch.tensor(stats.multivariate_normal.rvs(mean=mean_2[i], cov=cov_2[i], size = r))]) for i in range(N)]
    Y = torch.stack(Y)

    #Y = torch.abs(Y-mean_1.unsqueeze(1).expand(-1, 2*r, -1)) + mean_1.unsqueeze(1).expand(-1, 2*r, -1)

    C = C.unsqueeze(1).repeat(1, 2*r, 1)

    dataset = MyDataset(X.float(), C.float(), Y.float())
    return(dataset)