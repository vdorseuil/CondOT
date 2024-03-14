from torch.utils.data import Dataset, DataLoader
from scipy.stats import multivariate_normal
from gaussian_transport import compute_A, compute_w, gaussian_transport, get_gaussian_transport
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
    X = dataset.X.clone()
    C = dataset.C.clone()
    Y = dataset.Y.clone()

    X_gaussian = torch.ones_like(dataset.X)
    C_gaussian = torch.ones_like(dataset.C)
    Y_gaussian = torch.ones_like(dataset.Y)

    for i in range(X.shape[0]) :
        x_i = X[i]
        mean = torch.mean(x_i, dim = 0)
        cov_matrix = ((x_i-mean).T @ (x_i - mean)) / (x_i.shape[0] - 1)
        X_gaussian[i] = torch.tensor(multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=x_i.shape[0]))
    
    for i in range(Y.shape[0]) :
        y_i = Y[i]
        mean = torch.mean(y_i, dim = 0)
        cov_matrix = ((y_i-mean).T @ (y_i - mean)) / (y_i.shape[0] - 1)
        Y_gaussian[i] = torch.tensor(multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=y_i.shape[0]))
    
    for i in range(C.shape[0]) :
        C_gaussian[i] = C[i]

    gaussian_dataset = MyDataset(X_gaussian, C_gaussian, Y_gaussian)
    
    return(gaussian_dataset)

def get_gaussian_transport_dataset(gaussian_dataset) :
    X = gaussian_dataset.X.clone()
    C = gaussian_dataset.C.clone()
    Y = gaussian_dataset.Y.clone()

    l_transported = []
    for i in range(X.shape[0]) :
        x_i = X[i]
        y_i = Y[i]
        
        mean_x = torch.mean(x_i, dim = 0)
        cov_matrix_x = ((x_i-mean_x).T @ (x_i - mean_x)) / (x_i.shape[0] - 1)

        mean_y = torch.mean(y_i, dim = 0)
        cov_matrix_y = ((y_i-mean_y).T @ (y_i - mean_y)) / (y_i.shape[0] - 1)

        l_transported.append(get_gaussian_transport(X[i].unsqueeze(0), cov_matrix_x, cov_matrix_y, mean_x, mean_y)[0])
    
    Y_transport = torch.stack(l_transported, dim=0)
    X_transport = X.clone()
    C_transport = C.clone()

    gaussian_transport_dataset = MyDataset(X_transport, C_transport, Y_transport)
    
    return(gaussian_transport_dataset)