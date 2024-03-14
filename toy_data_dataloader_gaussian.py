from mydataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
import numpy as np
import torch

def generate_dataset(r = 100, N = 500) :
    mean_1_X = torch.tensor([0,0], dtype=torch.float32).repeat(N,1)
    mean_2_X = torch.tensor([2,0], dtype=torch.float32).repeat(N,1)
    cov_1_X = [0.2*torch.tensor(np.eye(2)) for i in range(N)]
    cov_2_X = [0.2*torch.tensor(np.eye(2)) for i in range(N)]
    X = [torch.concat([torch.tensor(stats.multivariate_normal.rvs(mean=mean_1_X[i], cov=cov_1_X[i], size = r)),torch.tensor(stats.multivariate_normal.rvs(mean=mean_2_X[i], cov=cov_2_X[i], size = r))]) for i in range(N)]
    X = torch.stack(X)

    mean_1_Y = torch.tensor([4,2], dtype=torch.float32).repeat(N,1)
    mean_2_Y = torch.tensor([2,4], dtype=torch.float32).repeat(N,1)
    cov_1_Y = [0.2*torch.tensor(np.eye(2)) for i in range(N)]
    cov_2_Y = [0.2*torch.tensor(np.eye(2)) for i in range(N)]
    Y = [torch.concat([torch.tensor(stats.multivariate_normal.rvs(mean=mean_1_Y[i], cov=cov_1_Y[i], size = r)),torch.tensor(stats.multivariate_normal.rvs(mean=mean_2_Y[i], cov=cov_2_Y[i], size = r))]) for i in range(N)]
    Y = torch.stack(Y)

    C = torch.cat([mean_1_Y, torch.stack(cov_1_Y).view(N,-1), mean_2_Y, torch.stack(cov_2_Y).view(N,-1)], dim=1)   
    C = C.unsqueeze(1).repeat(1, 2*r, 1)

    dataset = MyDataset(X.float(), C.float(), Y.float())
    return(dataset)