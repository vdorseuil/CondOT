from torch.utils.data import Dataset, DataLoader
from scipy.stats import multivariate_normal
from gaussian_transport import compute_A, compute_w, gaussian_transport
import torch

class MyDataset(Dataset):
    def __init__(self, X, C, Y):
        self.X = X
        self.C = C
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.Y[idx]
    
def get_gaussian_dataset(dataset) :
    X = dataset.X
    C = dataset.C
    Y = dataset.Y

    for i in range(X.shape[0]) :
        x_i = X[i]
        mean = torch.mean(x_i, dim = 0)
        cov_matrix = ((x_i-mean).T @ (x_i - mean)) / (x_i.shape[0] - 1)
        X[i] = torch.tensor(multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=x_i.shape[0]))
    
    for i in range(Y.shape[0]) :
        y_i = Y[i]
        mean = torch.mean(y_i, dim = 0)
        cov_matrix = ((y_i-mean).T @ (y_i - mean)) / (y_i.shape[0] - 1)
        Y[i] = torch.tensor(multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=y_i.shape[0]))
    
    return(dataset)

def get_gaussian_transport_dataset(gaussian_dataset) :
    X = gaussian_dataset.X
    C = gaussian_dataset.C
    Y = gaussian_dataset.Y

    X_transport = torch.ones_like(gaussian_dataset.X)
    C_transport = torch.ones_like(gaussian_dataset.C)
    Y_transport = torch.ones_like(gaussian_dataset.Y)

    for i in range(X.shape[0]) :
        x_i = X[i]
        y_i = Y[i]

        mean_x = torch.mean(x_i, dim = 0)
        cov_matrix_x = ((x_i-mean_x).T @ (x_i - mean_x)) / (x_i.shape[0] - 1)

        mean_y = torch.mean(y_i, dim = 0)
        cov_matrix_y = ((y_i-mean_y).T @ (y_i - mean_y)) / (y_i.shape[0] - 1)

        A = compute_A(cov_matrix_x, cov_matrix_y)
        w = compute_w(mean_x, mean_y, A)

        for j in range(x_i.shape[0]) :
            Y_transport[i,j] = gaussian_transport(x_i[j], A, w)
        
        X_transport[i] = X[i]
        C_transport[i] = C[i]

        gaussian_transport_dataset = MyDataset(X_transport, C_transport, Y_transport)
    
    return(gaussian_transport_dataset)