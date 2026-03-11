import numpy as np
import matplotlib.pyplot as plt

from QAOA import QAOA
from hamiltonians import ion_native_hamiltonian

from tqdm import tqdm

from scipy.interpolate import griddata


def get_landscape(Q,k=101):
    
    cost = np.zeros((k,k))
    beta = np.linspace(0,0.5*np.pi,k)
    gamma = np.linspace(0.0,2.0*np.pi,k)
    
    for i in range(k):
        for j in range(k):
            cost[i][j] = Q.expectation(angles=[gamma[i],beta[j]])
    return cost


def get_landscape_interp(Q,k=101,s=5):
    
    cost = np.zeros((s,s))
    beta = np.linspace(0,0.5*np.pi,s)
    gamma = np.linspace(0.0,2.0*np.pi,s)
    
    beta1 = np.linspace(0,0.5*np.pi,k)
    gamma1 = np.linspace(0.0,2.0*np.pi,k)
    
    points = []
    
    
    for i in range(s):
        for j in range(s):
            cost[i][j] = Q.expectation(angles=[gamma[i],beta[j]])
            points.append(np.array([gamma[i],beta[j]]))
            
    gamma_grid, beta_grid = np.meshgrid(gamma1,beta1)
            
    landscape = griddata(points,cost.reshape([s**2,]),(gamma_grid,beta_grid),method='cubic')
    
    return cost