import numpy as np
from scipy.optimize import minimize
from IPython.display import clear_output
import time

from hamiltonians import ion_native_hamiltonian

# =============================================================================
#                                 Класс QAOA
# =============================================================================

class QAOA:
    
    def __init__(self,depth,H1,H2):                          
        
        self.H1 = H1
        self.H2 = H2
        self.n = int(np.log2(int(len(self.H1)))) 
        
        #______________________________________________________________________________________________________
        self.X = self.new_mixerX()             

        #______________________________________________________________________________________________________
        
        self.min = min(self.H2)  
        self.max = max(self.H2)                
        self.deg = len(self.H2[self.H2 == self.min]) 
        self.p = depth                           
        
        self.heruistic_LW_seed1 = 10
        self.heruistic_LW_seed2 = 20
        
        #______________________________________________________________________________________________________   
    
    def new_mixerX(self):

        def split(x,k):
            return x.reshape((2**k,-1))
        def sym_swap(x):
            return np.asarray([x[-1],x[-2],x[1],x[0]])
        
        n = self.n
        x_list = []
        t1 = np.asarray([np.arange(2**(n-1),2**n),np.arange(0,2**(n-1))])
        t1 = t1.flatten()
        x_list.append(t1.flatten())
        t2 = t1.reshape(4,-1)
        t3 = sym_swap(t2)
        t1 = t3.flatten()
        x_list.append(t1)
        
        
        k = 1
        while k < (n-1):
            t2 = split(t1,k)
            t2 = np.asarray(t2)
            t1=[]
            for y in t2:
                t3 = y.reshape((4,-1))
                t4 = sym_swap(t3)
                t1.append(t4.flatten())
            t1 = np.asarray(t1)
            t1 = t1.flatten()
            x_list.append(t1)
            k+=1        
        
        return x_list
    #__________________________________________________________________________________________________________   
        
    def U_gamma(self,angle,state):
   
        t = -1j*angle
        state = state*np.exp(t*self.H1.reshape(2**self.n,1))
        
        return state
    
    def V_beta(self,angle,state):       
        
        c = np.cos(angle)
        s = np.sin(angle)
        
        for i in range(self.n):
            t = self.X[i]
            st = state[t]
            state = c*state + (-1j*s*st)
            
        return state

    #__________________________________________________________________________________________________________
    
    def qaoa_ansatz(self, angles):
        
        state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state 
    
    #__________________________________________________________________________________________________________ 
    
    def apply_ansatz(self, angles, state):

        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state
    
    #__________________________________________________________________________________________________________ 
    
    def expectation(self,angles): 

        state = self.qaoa_ansatz(angles)
        
        ex = np.vdot(state,state*(self.H2).reshape((2**self.n,1)))
        
        return np.real(ex)
    
   #__________________________________________________________________________________________________________ 
          
    def overlap(self,state):
        
        g_ener = min(self.H2)
        olap = 0
        for i in range(len(self.H2)):
            if self.H2[i] == g_ener:
                olap+= np.absolute(state[i])**2
        
        return olap
    
    #__________________________________________________________________________________________________________ 
    
    
    def sample_fidelities_fixed_depth(self, p, n_samples):
         
        F = np.zeros(n_samples)
        
        for s in range(n_samples):
            pars = np.random.uniform(0, 2*np.pi, 2*p)
            psi1 = self.qaoa_ansatz(pars)
                
            pars = np.random.uniform(0, 2*np.pi, 2*p)
            psi2 = self.qaoa_ansatz(pars)
                
            F[s] = (np.abs(np.dot(psi1.conj().T, psi2).item())**2)
                
        return F
    
    #__________________________________________________________________________________________________________ 
    
    def sample_fidelities(self, p_max, n_samples):

        fidelities = np.zeros((p_max, n_samples))
        
        for p in range(1, p_max + 1):
            self.p = p
            fidelities[p-1,:] = self.sample_fidelities_fixed_depth(p,n_samples)
                
        return fidelities
    
   #__________________________________________________________________________________________________________ 
    
    def get_cost_one_layer(self,obj,n_grid=25):
        
                
        cost = np.zeros((n_grid,n_grid))
        beta = np.linspace(0,0.5*np.pi,n_grid)
        gamma = np.linspace(0.0,2.0*np.pi,n_grid)
        
        # evaluate the energy matrix for a single-layered ansatz 
        for i in range(n_grid):
            for j in range(n_grid):
                cost[i][j] = obj([gamma[i],beta[j]])
                
        i0, j0 = np.unravel_index(cost.argmin(), cost.shape)
        opt = [gamma[i0], beta[j0]]  

        return cost, opt, [i0,j0]
            
   #__________________________________________________________________________________________________________ 
   
    def train_single_layer_matrix(self,obj,n_grid=25):

        cost, opt, ind = self.get_cost_one_layer(obj,n_grid=n_grid)
        initial_guess = opt     
        
        bds =  [(0.0,2*np.pi)] + [(0.0,0.5*np.pi)]
            
        res = minimize(obj,initial_guess,method='L-BFGS-B',\
                        jac=None, bounds=bds, options={'maxfun': 150000})
            
        opt_angles = res.x
        nfev = n_grid**2 + res.nfev
        
        return opt_angles, cost, nfev
    
    #__________________________________________________________________________________________________________ 
      
    def run_layerwise_training_interp(self):
        """
        Симуляция варационного QAOA-подобного квантового алгоритма с
        использованием эвристического метода оптимизации квантовой цепи 
        по слоям (layer-wise, LW).
        """
        
        # function for generating boundary conditions
        bds = lambda x: [(0.0,2*np.pi)]*x + [(0.0,0.5*np.pi)]*x
        
        t_start = time.time()
        nfev_tot = 0
        
        # train for a single layer
        opt_angles, cost, nfev = self.train_single_layer_matrix(self.expectation)
        nfev_tot += nfev
        
        # main loop w.r.t. the circuit depth
        while len(opt_angles) < 2*self.p:
            
            p1 = len(opt_angles)//2
            
            gamma = opt_angles[:p1]
            beta = opt_angles[p1:]
            
            gamma0 = np.zeros(p1+1)
            beta0 = np.zeros(p1+1)
            
            # INTERP heuristic
            for j in range(p1+1):
                gamma0[j] = j/p1 * gamma[j-1] + (p1-j)/p1 * gamma[j % p1]
                beta0[j] = j/p1 * beta[j-1] + (p1-j)/p1 * beta[j % p1]
            
            initial_guess = np.concatenate([gamma0, beta0], axis=0)
            
            # optimize all variational parameters simultaneously for the current circuit depth
            res = minimize(self.expectation,initial_guess,method='L-BFGS-B', jac=None,
                           bounds=bds(p1+1), options={'maxfun': 150000})    
            opt_angles = res.x
            nfev_tot += res.nfev
    
            
        # store the optimal variational parameters
        self.opt_angles = opt_angles    
            
        # store all simulation results
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)[0]
        self.r = (self.max - self.q_energy)/(self.max - self.min)
        self.nfev = nfev_tot
    
    #__________________________________________________________________________________________________________ 
    
    def train_single_layer(self,obj):
        
        temp = []
        nfev = 0
        
        for n_runs in range(self.heruistic_LW_seed1):
            
            initial_guess =  [np.random.uniform(0,2*np.pi)] +  [np.random.uniform(0,0.5*np.pi)]
            
            bds =  [(0.0,2*np.pi)] + [(0.0,0.5*np.pi)]
            
            res = minimize(obj,initial_guess,method='L-BFGS-B',\
                           jac=None, bounds=bds, options={'maxfun': 150000})
            temp.append([res.fun, res.x])
            nfev += res.nfev
            
        temp = np.asarray(temp,dtype=object)
        idx = np.argmin(temp[:,0])
        opt_angles = temp[idx][1]
        
        return opt_angles, nfev
    
    #__________________________________________________________________________________________________________ 
   
    def run_layerwise_training(self):
        """
        Симуляция варационного QAOA-подобного квантового алгоритма с
        использованием эвристического метода оптимизации квантовой цепи 
        по слоям (layer-wise, LW).
        """
        
        # function for generating boundary conditions
        bds = lambda x: [(0.0,2*np.pi)]*x + [(0.0,0.5*np.pi)]*x
        
        def combine(a,b):

            a = list(a)
            b = list(b)
            a1 = a[0:int(len(a)/2)]
            a2 = a[int(len(a)/2)::]
            b1 = b[0:int(len(b)/2)]
            b2 = b[int(len(b)/2)::]
            a = a1+b1
            b = a2+b2
            
            return a + b 
        
        t_start = time.time()
        nfev_tot = 0 # count total number of cost function evaluations
        
        # train for a single layer
        opt_angles, nfev = self.train_single_layer(self.expectation)
        nfev_tot += nfev
       
        # state vector
        t_state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        
        # main loop w.r.t. the circuit depth
        while len(opt_angles) < 2*self.p:
            
            t_state = self.qaoa_ansatz(opt_angles)
            ex = lambda x : np.real(np.vdot(self.apply_ansatz(x,t_state),
                                            self.apply_ansatz(x,t_state)*(self.H2).reshape((2**self.n,1))))
            
            lw_angles, nfev = self.train_single_layer(ex)
            nfev_tot += nfev
            
            opt_angles = combine(opt_angles,lw_angles)    
            
            # optimize all variational parameters simultaneously for the current circuit depth
            res = minimize(self.expectation,opt_angles,method='L-BFGS-B', jac=None,
                           bounds=bds(len(opt_angles)//2), options={'maxfun': 150000})    
            opt_angles = res.x
            nfev_tot += res.nfev
            
        # store the optimal variational parameters
        self.opt_angles = opt_angles    
            
        # store all simulation results
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(nfev)
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)[0]
        self.r = (self.max - self.q_energy)/(self.max - self.min)
        self.nfev = nfev_tot
    
     #__________________________________________________________________________________________________________ 
    
def run_QAOA_for_fixed_p(H1,H2,p,method,n_runs):
        
    # инициализация ионно-совместимого анзаца
    Q = QAOA(p,H1,H2)
       
    if method == 0:
        Q.run_layerwise_training_interp()
        return Q.q_energy, Q.opt_angles, Q.olap, Q.r, Q.nfev, Q.exe_time
    else:
        # инициализация массивов для хранения результатов запусков
        energies = np.zeros(n_runs)
        ovlp = np.zeros(n_runs)
        r = np.zeros(n_runs)
        angles = np.zeros((n_runs,2*p))
        nfev = np.zeros(n_runs, dtype=np.int64)
        exe_time = np.zeros(n_runs)
            
        # несколько запусков минимизации с помощью эвристического метода
        # оптимизации квантовой цепи по слоям в попытке достичь глобального минимума
        for i in range(n_runs):
            Q.run_layerwise_training()

            energies[i] = Q.q_energy
            angles[i,:] = Q.opt_angles
            ovlp[i] = Q.olap
            r[i] = Q.r
            nfev[i] = Q.nfev
            exe_time[i] = Q.exe_time
            
        # извлечение лучшего результата с наименьшей энергией
        imin = np.argmin(energies)
    
        return energies[imin], angles[imin,:], ovlp[imin], r[imin], np.sum(nfev), np.sum(exe_time)

def run_QAOA(H1,H2,p_max,method,n_runs):
    
    # инициализация массивов для хранения результатов запусков
    energies = np.zeros(p_max)
    ovlp = np.zeros(p_max)
    r = np.zeros(p_max)
    nfev = np.zeros(p_max,dtype=np.int64)
    exe_time = np.zeros(p_max)
    
    opt_angles = []
    
    # цикл глубине анзаца
    for p in range(1,p_max+1):
    
        energies[p-1], angles, ovlp[p-1], r[p-1], nfev[p-1], exe_time[p-1] = run_QAOA_for_fixed_p(H1,H2,p,method,n_runs)
        opt_angles.append(angles)

    return energies, opt_angles, ovlp, r, nfev, exe_time