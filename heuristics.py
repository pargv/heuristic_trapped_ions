import numpy as np
from scipy.optimize import minimize, minimize_scalar
from tqdm import tqdm
import time

from hamiltonians import *
from QAOA import QAOA, run_QAOA_for_fixed_p
from analysis import get_landscape

def cost_A(A,angles,n,coupling_mat,Q):
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q.H1 = H1
    return Q.expectation(angles=angles)

def cost_angles(angles,Q):
    return Q.expectation(angles=angles)

def train_variational_parameters(A,n,coupling_mat,Q):
    
    n_grid = 10
    cost_mat = np.zeros((n_grid,n_grid))
    
    beta = np.linspace(0,0.5*np.pi,n_grid)
    gamma = np.linspace(0.0,2.0*np.pi,n_grid)
    
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q.H1 = H1
    
    for i in range(n_grid):
        for j in range(n_grid):
            cost_mat[i][j] = Q.expectation(angles=[gamma[i],beta[j]])
            
    i0, j0 = np.unravel_index(cost_mat.argmin(), cost_mat.shape)
    x0 = [gamma[i0], beta[j0]]
    
    bounds = [(0.0,2*np.pi)] + [(0.0,0.5*np.pi)]          
    res = minimize(cost_angles,x0=x0,method='L-BFGS-B',bounds=bounds,
                    options={'maxiter': 1e6},args=(Q))
    
    return res.fun, res.x, res.nfev + n_grid**2

def train_controllable_parameters(angles,x0,n,coupling_mat,Q):
    
    bounds = [(-1.0,1.0)]*n
            
    res = minimize(cost_A,x0=x0,method='Powell',bounds=bounds,options={'maxiter': 1e6},args=(angles,n,coupling_mat,Q))
    return res.x, res.nfev

def run_heuristics(n,H2,coupling_mat,n_iter,gs,tol_lvl,max_restarts=20,eps=1e-3):
    
    # initial conditions
    A = np.random.uniform(0.1,0.8,size=n)*np.random.choice([-1,1],size=n)
    
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1,H1,H2)
    
    energy_prev = 10.0*np.abs(gs)
    energy = 0.0
    
    list_E = []
    list_A = []
    
    # loop over block coordinate descent iterations
    k = 0
    tot_iterations = 0
    count_restarts = 0
    nfev_tot = 0
    
    start_time = time.time()
    
    while(True):
        
        # optimization with respect to variational parameters 
        # with fixed controllable parameters
        energy, angles, nfev = train_variational_parameters(A,n,coupling_mat,Q)
        nfev_tot += nfev
        
        # check convergence criteria
        if np.abs(energy - gs) < tol_lvl*np.abs(gs):
            break
        
        # if we exceed the max allowed number of iterations or get stuck, restart from random A
        if k > n_iter or np.abs(energy - energy_prev) < eps:
            k = 0
            count_restarts += 1
            
            list_E.append(energy)
            list_A.append(A)
            
            # if we exceed max allowed number of restarts, terminate the algorithm
            if count_restarts > max_restarts:
                j = np.argmin(list_E)
                energy = list_E[j]
                A = list_A[j]
                break
            else:
                A = np.random.uniform(0.1,0.8,size=n)*np.random.choice([-1,1],size=n)
                energy_prev = 10.0*np.abs(gs)
                continue
        
        # optimization with respect to controllable parameters 
        # with fixed variational parameters
        A, nfev = train_controllable_parameters(angles,A,n,coupling_mat,Q)
        nfev_tot += nfev
        
        # update iteration counters
        k += 1
        tot_iterations += 1
        
        # save energy on the previous step
        energy_prev = energy
        
    exec_time = time.time() - start_time
    
    return A, energy, tot_iterations, exec_time, count_restarts, nfev_tot

def training(path,fname,n,weights,coupling_mat,n_iter,tol_lvl,max_restarts=20,eps=1e-3):
    
    # construct the S-K problem Hamiltonian
    H2 = get_hamiltonian(n,weights)
    
    # evaluate the exact ground state
    gs = np.min(H2)
    
    # optimize controllable parameters of the ion-native ansatz 
    A_trained, energy, tot_iterations, exec_time, count_restarts, nfev_tot = run_heuristics(n,H2,coupling_mat,n_iter,gs,tol_lvl,max_restarts,eps)
    
    save_training_data(path,fname,n,weights,gs,A_trained,energy,tot_iterations,exec_time,count_restarts,nfev_tot,tol_lvl)

def save_training_data(path,fname,n,weights,gs,A_trained,energy,tot_iterations,exec_time,count_restarts,nfev_tot,tol_lvl):
    
    half_energy = energy - gs < tol_lvl*np.abs(gs)
    
    file = path+fname
    with open(file,'w') as fid:
        fid.write("{0:2d}    ".format(n))
        fid.write("{0:8f}    ".format(gs))
        fid.write("{0:8f}    ".format(energy))
        fid.write("{0:1d}    ".format(half_energy))
        fid.write("{0:4d}     ".format(tot_iterations))
        fid.write("{0:8f}     ".format(exec_time))
        fid.write("{0:4d}     ".format(count_restarts))
        fid.write("{0:5d}     ".format(nfev_tot))
        fid.write("{0:4f}\n".format(tol_lvl))
        
        for w in weights:
            fid.write("{0:8f}  ".format(w))
        fid.write("\n")

        for i in range(n):
            fid.write("{0:8f}  ".format(A_trained[i]))
        fid.write("\n")
    
    fid.close()   
    
def get_trained_stats(path,fname,n_instances):
    
    avg_success = 0.0
    avg_iterations = 0.0
    avg_time = 0.0
    nfev = np.zeros(n_instances)
    
    i_bad = []
    
    for i in range(n_instances):
        file = path + fname + str(i+1) + '.txt'
        data = np.genfromtxt(file,skip_footer=2)
        
        if data[3]!=1:
            i_bad.append(i+1)
        avg_success += data[3]
        avg_iterations += data[4]
        avg_time += data[5]
        nfev[i] = data[-2]
    
    avg_success = avg_success/n_instances
    avg_iterations = avg_iterations/n_instances
    avg_time = avg_time/n_instances

    return avg_success, avg_iterations, avg_time, i_bad, nfev
    
def scaling_metric(alpha,n,A,H1,H2,c_gmin,coupling_mat,mu,k,eps):
    
    Q = QAOA(1,H1,H2)
    # scale controllable parameters
    if np.sum(np.abs(alpha*A) > 1.0):
        return 100
    
    H1_scaled = ion_native_hamiltonian(n,alpha*A,coupling_mat)
    Q.H1 = H1_scaled
    cost_mat = get_landscape(Q,k=k)
    
    c_min = np.min(cost_mat)
    
    if np.abs(c_min - c_gmin) > eps:
        return 1
    
    return -cost_mat[cost_mat < mu*c_min].size/cost_mat.size

def rescale_contr_params(n,A,H2,coupling_mat,mu,k=20,eps=0.05):
    
    # estimate global minimum
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1,H1,H2)
    cost_mat = get_landscape(Q,k=k)
    c_gmin = np.min(cost_mat)
    
    res = minimize(scaling_metric,x0=0.8,bounds=(0.0,1.0),method='COBYLA',args=(n,A,H1,H2,c_gmin,coupling_mat,mu,k,eps))
    alpha = res.x[0]
    A_rescaled = alpha*A
    
    return A_rescaled, alpha, res.nfev*k**2

def evaluation_metric(fout,fdata,n,coupling_mat,method,n_runs,mu):
        
    # calculate the problem Hamiltonian based on weights saved in a datafile
    weights = list(np.genfromtxt(fdata,skip_header=1,skip_footer=1))
    H2 = get_hamiltonian(n,weights)

    gs = np.min(H2)
    emax = np.max(H2)
    
    e = np.unique(np.sort(H2))
    gap = e[1] - e[0]
            
    # read the trained controllable parameters
    A_trained = np.genfromtxt(fdata,skip_header=2)
        
    # rescale the trained controllable parameters
    A_rescaled, alpha, nfev_rscl = rescale_contr_params(n,A_trained,H2,coupling_mat,mu)
        
    # evaluate the QAOA energy at p = n layers
    p = n
    H1 = ion_native_hamiltonian(n,A_rescaled,coupling_mat)
    energy, angles, ovlp, r, nfev, exe_time = run_QAOA_for_fixed_p(H1,H2,p,method,n_runs)

    with open(fout,'w') as fid:
        fid.write("{0:2d}    ".format(n))
        fid.write("{0:2d}     ".format(p))
        fid.write("{0:8f}    ".format(gs))
        fid.write("{0:8f}    ".format(emax))
        fid.write("{0:8f}    ".format(energy))
        fid.write("{0:8f}    ".format(gap))
        fid.write("{0:8f}    ".format(ovlp))
        fid.write("{0:8f}    ".format(exe_time))
        fid.write("{0:8f}    ".format(alpha))
        fid.write("{0:5d}    ".format(nfev_rscl))
        fid.write("{0:5d}\n".format(nfev))
        
        for w in weights:
            fid.write("{0:8f}  ".format(w))
        fid.write("\n")

        for i in range(n):
            fid.write("{0:8f}  ".format(A_rescaled[i]))
        fid.write("\n")
        
        for i in range(2*p):
            fid.write("{0:8f}    ".format(angles[i]))
        fid.write("\n")
    
    fid.close()
    
    
def layerwise_evaluation(fout,fdata,n,coupling_mat,p_max,method,n_runs):
        
    # calculate the problem Hamiltonian based on weights saved in a datafile
    data = np.genfromtxt(fdata,skip_footer=3)
    weights = list(np.genfromtxt(fdata,skip_header=1,skip_footer=2))
    H2 = get_hamiltonian(n,weights)

    gs = data[2]
    emax = data[3]
    gap = data[5]
            
    # read the trained controllable parameters
    A = np.genfromtxt(fdata,skip_header=2,skip_footer=1)
        
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    
    # evaluate the QAOA energy up to p_max layers
    
    energies = np.zeros(p_max)
    ovlp = np.zeros(p_max)
    r = np.zeros(p_max)
    nfev = np.zeros(p_max)
    exe_time = np.zeros(p_max)
    opt_angles = []
    
    for p in range(p_max):
        
        energies[p], angles, ovlp[p], r[p], nfev[p], exe_time[p] = run_QAOA_for_fixed_p(H1,H2,p+1,method,n_runs)
        opt_angles.append(angles)
    
    with open(fout,'w') as fid:
        fid.write("{0:2d}    ".format(n))
        fid.write("{0:8f}    ".format(gs))
        fid.write("{0:8f}    ".format(emax))
        fid.write("{0:8f}\n".format(gap))
        
        for w in weights:
            fid.write("{0:8f}  ".format(w))
        fid.write("\n")

        for i in range(n):
            fid.write("{0:8f}  ".format(A[i]))
        fid.write("\n")
        
        for p in range(p_max):
            fid.write("{0:2d}    ".format(p+1))
            fid.write("{0:8f}    ".format(energies[p]))
            fid.write("{0:8f}    ".format(ovlp[p]))
            fid.write("{0:10f}   ".format(r[p]))
            fid.write("{0:10f}    ".format(nfev[p]))
            fid.write("{0:10f}\n".format(exe_time[p]))
            
        for p in range(p_max):
            angles = opt_angles[p]
            for i in range(2*(p+1)):
                fid.write("{0:8f}    ".format(angles[i]))
            fid.write("\n")
    
    fid.close()
    
def layerwise_evaluation_fixed_A(fout,fdata,n,coupling_mat,A,p_max,method,n_runs):
        
    # calculate the problem Hamiltonian based on weights saved in a datafile
    data = np.genfromtxt(fdata,skip_footer=3)
    weights = list(np.genfromtxt(fdata,skip_header=1,skip_footer=2))
    H2 = get_hamiltonian(n,weights)

    gs = data[2]
    emax = data[3]
    gap = data[5]
        
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    
    # evaluate the QAOA energy up to p_max layers
    
    energies = np.zeros(p_max)
    ovlp = np.zeros(p_max)
    r = np.zeros(p_max)
    nfev = np.zeros(p_max)
    exe_time = np.zeros(p_max)
    opt_angles = []
    
    for p in range(p_max):
        
        energies[p], angles, ovlp[p], r[p], nfev[p], exe_time[p] = run_QAOA_for_fixed_p(H1,H2,p+1,method,n_runs)
        opt_angles.append(angles)
    
    with open(fout,'w') as fid:
        fid.write("{0:2d}    ".format(n))
        fid.write("{0:8f}    ".format(gs))
        fid.write("{0:8f}    ".format(emax))
        fid.write("{0:8f}\n".format(gap))
        
        for w in weights:
            fid.write("{0:8f}  ".format(w))
        fid.write("\n")

        for i in range(n):
            fid.write("{0:8f}  ".format(A[i]))
        fid.write("\n")
        
        for p in range(p_max):
            fid.write("{0:2d}    ".format(p+1))
            fid.write("{0:8f}    ".format(energies[p]))
            fid.write("{0:8f}    ".format(ovlp[p]))
            fid.write("{0:10f}   ".format(r[p]))
            fid.write("{0:10f}    ".format(nfev[p]))
            fid.write("{0:10f}\n".format(exe_time[p]))
            
        for p in range(p_max):
            angles = opt_angles[p]
            for i in range(2*(p+1)):
                fid.write("{0:8f}    ".format(angles[i]))
            fid.write("\n")
    
    fid.close()

def layerwise_evaluation_qaoa(fout,fdata,n,p_max,method,n_runs):
        
    # calculate the problem Hamiltonian based on weights saved in a datafile
    data = np.genfromtxt(fdata,skip_footer=3)
    weights = list(np.genfromtxt(fdata,skip_header=1,skip_footer=2))
    H2 = get_hamiltonian(n,weights)

    gs = data[2]
    emax = data[3]
    gap = data[5]
    
    # evaluate the QAOA energy up to p_max layers
    
    energies = np.zeros(p_max)
    ovlp = np.zeros(p_max)
    r = np.zeros(p_max)
    nfev = np.zeros(p_max, dtype=np.int64)
    exe_time = np.zeros(p_max)
    opt_angles = []
    
    for p in range(p_max):
        
        energies[p], angles, ovlp[p], r[p], nfev[p], exe_time[p] = run_QAOA_for_fixed_p(H2,H2,p+1,method,n_runs)
        opt_angles.append(angles)
    
    with open(fout,'w') as fid:
        fid.write("{0:2d}    ".format(n))
        fid.write("{0:8f}    ".format(gs))
        fid.write("{0:8f}    ".format(emax))
        fid.write("{0:8f}\n".format(gap))
        
        for w in weights:
            fid.write("{0:8f}  ".format(w))
        fid.write("\n")
        
        for p in range(p_max):
            fid.write("{0:2d}    ".format(p+1))
            fid.write("{0:8f}    ".format(energies[p]))
            fid.write("{0:8f}    ".format(ovlp[p]))
            fid.write("{0:10f}   ".format(r[p]))
            fid.write("{0:7d}    ".format(nfev[p]))
            fid.write("{0:10f}\n".format(exe_time[p]))
            
        for p in range(p_max):
            angles = opt_angles[p]
            for i in range(2*(p+1)):
                fid.write("{0:8f}    ".format(angles[i]))
            fid.write("\n")
            
    fid.close()
    
    
def get_evaluation_stats(path,fname,n_instances,eps):
    
    avg_success = 0.0
    violated_A = 0
    avg_ar = 0.0
    nfev_rscl = np.zeros(n_instances)
    nfev = np.zeros(n_instances)
    
    i_bad = []
    
    for i in range(n_instances):
        file = path + fname + str(i+1) + '.txt'
        data = np.genfromtxt(file,skip_footer=3)
        
        avg_ar += (data[4] - data[3])/(data[2] - data[3])
        nfev_rscl[i] = data[-2]
        nfev[i] = data[-1]
        
        if data[6] > eps:
            success = 1
        else:
            success = 0
            i_bad.append(i+1)
            
        avg_success += success
        
        A = np.genfromtxt(file,skip_footer=1,skip_header=2)
        violated_A += (np.sum(np.abs(A) > 1.0) > 0)
    
    avg_success = avg_success/n_instances
    avg_ar = avg_ar/n_instances
    avg_nfev = np.mean(nfev)
    std_nfev = np.std(nfev)
    
    return avg_success, avg_ar, nfev_rscl, avg_nfev, std_nfev, violated_A, i_bad

def get_lw_stats(path,fname,p_max,n_instances,eps=0.5,s=0):
    
    r = np.zeros((p_max,n_instances))
    frac_solved = np.zeros(p_max)
    ovlp = np.zeros((p_max,n_instances))
    nfev = np.zeros((p_max,n_instances))
    
    i_bad = []
    
    for i in range(n_instances):
        file = path + fname + str(i+1) + '.txt'
        
        if s==0:
            data = np.genfromtxt(file,skip_header=3,skip_footer=p_max)
        
            r[:,i] = 1 - data[:,3]
            ovlp[:,i] = data[:,2]
            nfev[:,i] = data[:,4]
        
        else:
            data = np.genfromtxt(file,skip_header=2,skip_footer=p_max)
        
            r[:,i] = 1 - data[:,3]
            ovlp[:,i] = data[:,2]
        
        frac_solved += data[:,2] > eps
        if data[3,2] <= eps:
            i_bad.append(i+1)
    
    avg_r = np.mean(r,axis=1)
    std_r = np.std(r,axis=1)
    avg_ovlp = np.mean(ovlp,axis=1)
    std_ovlp = np.std(ovlp,axis=1)
    avg_nfev = np.mean(nfev,axis=1)
    std_nfev = np.std(nfev,axis=1)
    frac_solved = frac_solved/n_instances
    
    return avg_r, std_r, avg_ovlp, std_ovlp,  avg_nfev, std_nfev, frac_solved, i_bad

def get_lw_stats_ind(path,fname,p_max,indices):
    
    n_instances = len(indices)
    r = np.zeros((p_max,n_instances))
    ovlp = np.zeros((p_max,n_instances))
    
    for k, i in enumerate(indices):
        file = path + fname + str(i) + '.txt'
        
        data = np.genfromtxt(file,skip_header=3,skip_footer=p_max)
        
        r[:,k] = data[:,3]
        ovlp[:,k] = data[:,2]
    
    avg_r = np.mean(r,axis=1)
    std_r = np.std(r,axis=1)
    avg_ovlp = np.mean(ovlp,axis=1)
    std_ovlp = np.std(ovlp,axis=1)
    
    return avg_r, std_r, avg_ovlp, std_ovlp