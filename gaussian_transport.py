import scipy.linalg
import torch
import numpy as np
import sys

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
    A =(np.array(A) + np.eye(A.shape[0]) * 1e-5).astype('float32') #regulzarization

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
    u = torch.tensor(u).float()
    A = torch.tensor(A).float()
    w = torch.tensor(w).float()
  
    vect = A@(u - w)

    return ((1/2) * torch.dot(vect.real, vect.real))

def gaussian_transport_data(source, target, data) :
    source = torch.tensor(source)
    target = torch.tensor(target)
    data = torch.tensor(data)

    vect = torch.zeros_like(data)
    for i in range(source.shape[0]):
        mean_source = torch.mean(source[i], dim = 0)
        mean_target = torch.mean(target[i], dim = 0)

        cov_source = ((source[i]-mean_source).T @ (source[i] - mean_source)) / (source[i].shape[0] - 1)
        cov_target = ((target[i]-mean_target).T @ (target[i] - mean_target)) / (target[i].shape[0] - 1)

        # mean_source = mean_source.detach().numpy()
        # mean_target = mean_target.detach().numpy()
        # cov_source = cov_source.detach().numpy()
        # cov_target = cov_target.detach().numpy()



        A = compute_A(cov_source, cov_target)
        w = compute_w(mean_source, mean_target, A)

        A = torch.tensor(A)
        w = torch.tensor(w)


        for j in range(source[i].shape[0]) :
            vect[i,j] = gaussian_transport(source[i][j], A, w)
    
        #vec = A@(data[i]- w)
        #vect[i] = (1/2) * torch.dot(vec, vec)   

    return (vect)