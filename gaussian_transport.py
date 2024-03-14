import scipy.linalg
import torch
import numpy as np
import sys

def get_mean(batch):
    means = torch.mean(batch, dim=1)
    average_mean = torch.mean(means, dim=0)
    return(average_mean)

def get_covariance(batch):
    n = batch.size(1) - 1
    mean = torch.mean(batch, dim=1, keepdim=True)
    batch = batch - mean  # Centering the data
    cov = torch.matmul(batch.transpose(1, 2), batch) / n
    return(torch.mean(cov, dim=0))

def compute_A(cov1, cov2) :
    cov1 = np.array(cov1)
    cov2 = np.array(cov2)

    cov1_moins12 = scipy.linalg.fractional_matrix_power(cov1, -0.5)
    cov1_12 = scipy.linalg.sqrtm(cov1)
    
    A = scipy.linalg.sqrtm(cov1_moins12 @ scipy.linalg.sqrtm(cov1_12 @ cov2 @ cov1_12) @ cov1_moins12)
    A = np.real(A)

    return(A)

def compute_w(m1, m2, A) :
    m1 = np.array(m1)
    m2 = np.array(m2)
    A = (np.array(A) + np.eye(A.shape[0]) * 1e-5).astype('float32') #regulzarization A to guarantee invertibility

    #test if A is invertible
    if np.linalg.det(A) == 0:
        sys.exit('A is not invertible') 

    w = m1 - np.linalg.inv((np.transpose(A)@A) +  np.eye(A.shape[0]) * 1e-5)@m2
    return(w)

def compute_b(m1, m2, A) :
    m1 = np.array(m1)
    m2 = np.array(m2)
    A = np.array(A)

    b = m2 - np.transpose(A)@A@m1
    return(b)

def t(A,b):
    A = np.array(A)
    b = np.array(b)
    
    t = np.transport(b)@np.linalg.inv(np.transpose(A)@A)@b
    return(t)

def gaussian_transport(u,A,w):
    A = torch.tensor(A).float()
    w = torch.tensor(w).float()
    if len(u.shape) == 1:
        vect = A@(u-w)
        transported = (1/2) * torch.dot(vect.real, vect.real) #sum((A@(u-w))^2)
    else : 
        u_minus_w = u - w
        vect = torch.einsum('ij,abj->abi', A, u_minus_w) #A@(u-w)
        transported = (1/2) * torch.sum(vect.real * vect.real, dim=-1, keepdim=True) #sum((A@(u-w))^2)
    return (transported)

def get_gaussian_transport(u, cov1, cov2, m1, m2) :
    A = compute_A(cov1, cov2)
    w = compute_w(m1, m2, A)
    return(gaussian_transport(u, A, w))