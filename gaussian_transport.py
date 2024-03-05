import scipy.linalg
import torch
import numpy as np

def compute_A(cov1, cov2) :
    cov1 = cov1
    cov2 = cov2

    cov1_moins12 = scipy.linalg.fractional_matrix_power(cov1, -0.5)
    cov1_12 = scipy.linalg.sqrtm(cov1)
    
    A = scipy.linalg.sqrtm(cov1_moins12 @ scipy.linalg.sqrtm(cov1_12 @ cov2 @ cov1_12) @ cov1_moins12)

    return(A)

def compute_w(m1, m2, A) :
    w = m1 - np.linalg.inv(np.transpose(A)@A)@m2
    return(w)

def compute_b(  m1, m2, A) :
    b = m2 - np.transpose(A)@A@m1
    return(b)

def t(A,b):
    t = np.transport(b)@np.linalg.inv(np.transpose(A)@A)@b
    return(t)

def gaussian_transport(u,A,w):
    u = torch.tensor(u)
    A = torch.tensor(A)
    w = torch.tensor(w)
  
    vect = A@(u - w)
    return ((1/2) * torch.dot(vect, vect) )