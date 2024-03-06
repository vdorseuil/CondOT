from mydataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
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
    locs = locs + torch.tensor(stats.norm.rvs(loc=0, scale=1, size=(N, r)), dtype=torch.float32) #bruit gaussien
    
    scales = torch.abs(torch.tensor(stats.norm.rvs(loc=0.9, scale=1, size=(N, r)), dtype=torch.float32)) + 0.1

    locs2D = locs.unsqueeze(-1).expand(-1, -1, d)
    scales2D = scales.unsqueeze(-1).expand(-1, -1, d)

    Y = torch.tensor(stats.norm.rvs(loc=locs2D, scale=scales2D, size=(N, r, d)), dtype=torch.float32)
    Y = torch.abs(Y)

    C = torch.stack((locs, scales), dim = 2)

    dataset = MyDataset(X, C, Y)

    return(dataset)