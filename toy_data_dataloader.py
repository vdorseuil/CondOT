from mydataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats

import torch



def get_dataloader(d = 2, r = 100, N = 500, batch_size = 500):
    d = d # dimension of the data
    r = r # number of points to simulate my gaussian
    N = N # number of samples


    # simulate the gaussian
    X = torch.tensor(stats.norm.rvs(loc=0, scale=1, size=(N, r, d)), dtype=torch.float32)
    locs = torch.randint(-10, 10, (N,), dtype=torch.float32).repeat_interleave(r).view(N, r)
    scales = 5*torch.rand(N, dtype=torch.float32).repeat_interleave(r).view(N, r)

    C = torch.stack((locs, scales), dim = 2)

    locs2D = locs.unsqueeze(-1).expand(-1, -1, d)
    scales2D = scales.unsqueeze(-1).expand(-1, -1, d)

    Y = torch.tensor(stats.norm.rvs(loc=locs2D, scale=scales2D, size=(N, r, d)), dtype=torch.float32)

    # print(locs.size())
    # print(X.size())
    # print(C.size())
    # print(Y.size())


    dataset = MyDataset(X, C, Y)
    batch_size = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
