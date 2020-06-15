
import numpy as np
import matplotlib.pyplot as plt
import torch
import sympy
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr

def getKernel(diff_order,acc_order=2,dim=None,scheme='central'):
    
    def central(diff_order,acc_order,dim):
        '''
        input: order of the derivative and the dimension (It will round it to next odd)
        returns a kernel for a given order.
        default accuracy order is 2
        output index order  .. xi-2 xi-1 xi xi+1 xi+2 ..
        '''
        #minium dim to be 3 it will return an error if the order and the dimension not matches.
        n_eles = diff_order+acc_order-1
        if dim==None:
            dim = n_eles

        if dim<n_eles:
            raise("The dim should be >- than the diff+acc-1")


        kernel = np.zeros(dim)

        B = np.zeros(n_eles)
        B[diff_order] = 1
        A = np.zeros((n_eles,n_eles))
        #first row by 1
        A[0] = np.ones(n_eles)

        numerators = np.zeros(n_eles)
        seq = np.arange(np.floor(n_eles/2))+1
        numerators[:int(np.floor(n_eles/2))] = seq[::-1]
        numerators[int(np.floor((n_eles-1)/2)+1):] = seq
        #let's fill the value of this matrix row wise 
        for i in range(1,n_eles,1):
            fact = np.math.factorial(i) 
            row = []
            prefix = 1
            for j in range(n_eles):
                if j>np.floor((n_eles-1)/2):
                    prefix = (-1)**i

                row.append(prefix*(numerators[j]**i)/fact)
            A[i] = row
        soln = np.linalg.inv(A).dot(B)

        soln = soln[::-1]
        len_sol = len(soln)
        if len_sol%2==0:
            a,b = np.split(soln,2)
            soln = np.zeros(len_sol+1)
            soln[:int(len_sol/2)] = a
            soln[int(len_sol/2)+1:] = b
            len_sol += 1

        kernel = np.pad(soln,int((dim-len_sol)/2))
        return kernel  
    
    
    def right(diff_order,acc_order,dim):
        '''
        output index order  xi xi+1 xi+2 .. ..
        '''
        n_eles = diff_order+acc_order
        B = np.zeros(n_eles)
        B[diff_order] = 1
        A = np.zeros((n_eles,n_eles))

        first_col = np.zeros(n_eles)
        first_col[0] = 1
        A[:,0] = first_col
        A[0] = np.ones(n_eles)

        for i in range(1,n_eles,1):
            for j in range(1,n_eles):
                A[i][j] = (j**i)/np.math.factorial(i) 

        kernel = np.linalg.inv(A).dot(B)
        len_ker = len(kernel)
        
        return kernel
    
    def left(diff_order,acc_order,dim):
        '''
        output index order  xi-2 xi-i xi .. ..
        '''
        n_eles = diff_order+acc_order
        B = np.zeros(n_eles)
        B[diff_order] = 1

        A = np.zeros((n_eles,n_eles))
        A[0] = np.ones(n_eles)        
        first_col = np.zeros(n_eles)
        first_col[0] = 1
        A[:,0] = first_col


        for i in range(1,n_eles,1):
            for j in range(1,n_eles):
                A[i][j] = ((-j)**i)/np.math.factorial(i) 

        kernel = np.linalg.inv(A).dot(B)
        return kernel[::-1]
    
    if scheme=='central':
        return central(diff_order,acc_order,dim)
    elif scheme=='left':
        return left(diff_order,acc_order,dim)
    elif scheme=='right':
        return right(diff_order,acc_order,dim)
        

        
def padInput(X,diff_order,acc_order=2,dim=None):
    '''
        pads the reqruied amount in the start and the end
    '''
    if dim is None:
        dim = diff_order+acc_order-1
    def padStart(X,diff_order,acc_order):
        central_kernel = getKernel(diff_order,acc_order,scheme='central')
        right_kernel = getKernel(diff_order,acc_order,scheme='right')
        n_c = len(central_kernel)
        n_r = len(right_kernel)
        n_eles = diff_order+acc_order-1 #central kernel
        n_pad = int(np.floor(n_eles/2))

        index = int(np.floor(n_c/2))-1
        for i in range(n_pad):
            right_conv = np.correlate(X[index:index+n_r],right_kernel)
            central_conv = np.correlate(X[0:n_c-1],central_kernel[1:])
            x_prev = (right_conv-central_conv)*1.0/central_kernel[0]
            X = np.concatenate((x_prev,X))
        return X

    def padEnd(X,diff_order,acc_order):
        central_kernel = getKernel(diff_order,acc_order,scheme='central')
        left_kernel = getKernel(diff_order,acc_order,scheme='left')
        n_c = len(central_kernel)
        n_l = len(left_kernel)
        n_eles = diff_order+acc_order-1 #central kernel
        n_pad = int(np.floor(n_eles/2))
        index = len(X) - (int(np.floor(n_c/2))-1)
        
        for i in range(n_pad):
            
            left_conv = np.correlate(X[index-n_l:index],left_kernel)
            central_conv = np.correlate(X[-(n_c-1):],central_kernel[:-1])
            x_next = (left_conv-central_conv)*1.0/central_kernel[-1]
            X = np.concatenate((X,x_next))
            index += 1
        return X
    
    X = padStart(X,diff_order,acc_order)
    X = padEnd(X,diff_order,acc_order)
    
    actual_dim = diff_order+acc_order-1
    if actual_dim%2 == 0:
        actual_dim += 1
    X = np.pad(X,int((dim-actual_dim)/2))
    return X



class TimeStepperSingle:
    def __init__(self,dx,dt,diff_coeff,time_scheme='rk4',acc_order=2):
        self.dx = dx
        self.dt = dt
        self.diff_coeff = diff_coeff
        self.acc_order = acc_order
        self.time_scheme = time_scheme
        self.kernel = getKernel(2,acc_order)
    def _step_(self, ut):
        dt = self.dt
        if self.time_scheme == 'rk4': # classical 4-stage 4th-order Runge–Kutta method
            K1 = self.RightHandItems(ut)
            K2 = self.RightHandItems(ut+dt/2*K1)
            K3 = self.RightHandItems(ut+dt/2*K2)
            K4 = self.RightHandItems(ut+dt*K3)
            rhi = dt/6*(K1+2*K2+2*K3+K4)
        elif self.time_scheme == 'rk2': # 2-stage 2nd-order Runge–Kutta method
            K1 = self.RightHandItems(ut)
            K2 = self.RightHandItems(ut+dt*K1)
            rhi = dt/2*(K1+K2)
        else: # forward Euler
            rhi = dt*self.RightHandItems(ut)
        return rhi
    
    def step(self, ut):
        ut_1 = ut+self._step_(ut)
        return ut_1

    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            F = self.diff_coeff/self.dx**2
            ut_padded = padInput(ut,2,self.acc_order)
            uxx_t = np.correlate(ut_padded,self.kernel)    
            return F*uxx_t

        



class DataModelSingle:
    def __init__(self,data_name,Nt,Nx,dt,dx):
        self.data_name = data_name
        self.Nt = Nt
        self.Nx = Nx
        self.dt = dt
        self.dx = dx
    
    @property
    def Lx(self):
        return self.dx*self.Nx
    
    @property
    def Lt(self):
        return self.dt*self.Nt
    
    def data(self):
        '''returns data
        '''
        raise NotImplementedError
        
    def _getMeshPoints(self):
        x = np.linspace(0,self.Lx,self.Nx)
        t = np.linspace(0,self.Lt,self.Nt)
        return x,t

    

    

    
class DiffusionDataSingle(DataModelSingle,TimeStepperSingle):
    def __init__(self,data_name,Nt,Nx,dt,dx,diff_coeff,acc_order=2,sigma=0.05,batch_size = 1,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        super(DiffusionDataSingle,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.sigma = sigma
        self.time_scheme = time_scheme
        self.diff_coeff = diff_coeff
        self.acc_order = acc_order
        self.kernel = getKernel(2,acc_order)
#         self.ts = TimeStepper(self.dx,self.dt,time_scheme=time_scheme)
        self._data = self._generateData()

    
    def _generateData(self):
        self.x,self.t =  self._getMeshPoints()
        u0 = self._IGaussian()
        u0[0] = u0[-1] = 0
        u0 = (u0-u0.min())/(u0.max()-u0.min())
        u = []   
        u_viz = np.zeros((self.Nt,self.Nx)) #u(t,x)         
        ut = u0
        u_viz[0] = u0
        u0 = np.reshape(u0,(1,1,-1))
        u.append(torch.from_numpy(u0))
        for i in range(1,self.Nt):
            u_temp = np.zeros(self.Nx)
            u.append(u_temp)
            u[i] = self.step(ut)
            #boundary condtion
            u[i][0] = 0
            u[i][-1] = 0
            u_viz[i] = u[i]
            ut = u[i]
            u[i] = torch.from_numpy(np.reshape(ut,(1,1,-1)))
        self.u_viz = u_viz
        return u
        
    def _IGaussian(self):
        return np.exp(-0.5*((self.x-self.Lx/2.0)**2)/self.sigma**2)
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            F = self.diff_coeff/self.dx**2
            ut_padded = padInput(ut,2,self.acc_order)
            uxx_t = np.correlate(ut_padded,self.kernel)   
            return F*uxx_t

    
        
class BurgersEqnSingle(DataModelSingle,TimeStepperSingle):
    def __init__(self,data_name,Nt,Nx,dt,dx,viscosity,acc_order=2,sigma=0.05,batch_size = 1,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        super(BurgersEqnSingle,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.sigma = sigma
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
#         self.ts = TimeStepper(self.dx,self.dt,time_scheme=time_scheme)
        self._data = self._generateData()   
    
        
    def _generateData(self):
        self.x,self.t =  self._getMeshPoints()
        u0 = self._IGaussian()
        u0[0] = u0[-1] = 0
        u0 = (u0-u0.min())/(u0.max()-u0.min())
        u_viz = np.zeros((self.Nt,self.Nx)) #u(t,x)         
        u = []   
        ut = u0
        u_viz[0] = u0
        u0 = np.reshape(u0,(1,1,-1))
        u.append(torch.from_numpy(u0))
        for i in range(1,self.Nt):
            u_temp = np.zeros(self.Nx)
            u.append(u_temp)
            u[i] = self.step(ut)
            #boundary condtion
            u[i][0] = 0
            u[i][-1] = 0
            u_viz[i] = u[i]
            ut = u[i]
            u[i] = torch.from_numpy(np.reshape(ut,(1,1,-1)))
        
        self.u_viz = u_viz
        return u
        
    def _IGaussian(self):
        return np.exp(-0.5*((self.x-self.Lx/2.0)**2)/self.sigma**2)
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            #1st order
            kernel_or1 = getKernel(1,self.acc_order)
            ut_padded_or1 = padInput(ut,1,self.acc_order)
            ux_t = np.correlate(ut_padded_or1,kernel_or1)/self.dx     
            #2nd order 
            kernel_or2 = getKernel(2,self.acc_order)
            ut_padded_or2 = padInput(ut,2,self.acc_order)
            uxx_t = np.correlate(ut_padded_or2,kernel_or2)/self.dx**2
            
            return self.viscosity*uxx_t - ut*ux_t
        



##################### For torch/Batch data #########################

def getKernelTorch(diff_order,acc_order=2,dim=None,scheme='central'):
    
    def central(diff_order,acc_order,dim):
        '''
        input: order of the derivative and the dimension (It will round it to next odd)
        returns a kernel for a given order.
        default accuracy order is 2
        output index order  .. xi-2 xi-1 xi xi+1 xi+2 ..
        '''
        #minium dim to be 3 it will return an error if the order and the dimension not matches.
        n_eles = diff_order+acc_order-1
        if dim==None:
            dim = n_eles

        if dim<n_eles:
            raise("The dim should be >- than the diff+acc-1")


        kernel = np.zeros(dim)

        B = np.zeros(n_eles)
        B[diff_order] = 1
        A = np.zeros((n_eles,n_eles))
        #first row by 1
        A[0] = np.ones(n_eles)

        numerators = np.zeros(n_eles)
        seq = np.arange(np.floor(n_eles/2))+1
        numerators[:int(np.floor(n_eles/2))] = seq[::-1]
        numerators[int(np.floor((n_eles-1)/2)+1):] = seq
        #let's fill the value of this matrix row wise 
        for i in range(1,n_eles,1):
            fact = np.math.factorial(i) 
            row = []
            prefix = 1
            for j in range(n_eles):
                if j>np.floor((n_eles-1)/2):
                    prefix = (-1)**i

                row.append(prefix*(numerators[j]**i)/fact)
            A[i] = row
        soln = np.linalg.inv(A).dot(B)

        soln = soln[::-1]
        len_sol = len(soln)
        if len_sol%2==0:
            a,b = np.split(soln,2)
            soln = np.zeros(len_sol+1)
            soln[:int(len_sol/2)] = a
            soln[int(len_sol/2)+1:] = b
            len_sol += 1

        kernel = np.pad(soln,int((dim-len_sol)/2))
        return kernel  
    
    
    def right(diff_order,acc_order,dim):
        '''
        output index order  xi xi+1 xi+2 .. ..
        '''
        n_eles = diff_order+acc_order
        B = np.zeros(n_eles)
        B[diff_order] = 1
        A = np.zeros((n_eles,n_eles))

        first_col = np.zeros(n_eles)
        first_col[0] = 1
        A[:,0] = first_col
        A[0] = np.ones(n_eles)

        for i in range(1,n_eles,1):
            for j in range(1,n_eles):
                A[i][j] = (j**i)/np.math.factorial(i) 

        kernel = np.linalg.inv(A).dot(B)
        len_ker = len(kernel)
        
        return kernel
    
    def left(diff_order,acc_order,dim):
        '''
        output index order  xi-2 xi-i xi .. ..
        '''
        n_eles = diff_order+acc_order
        B = np.zeros(n_eles)
        B[diff_order] = 1

        A = np.zeros((n_eles,n_eles))
        A[0] = np.ones(n_eles)        
        first_col = np.zeros(n_eles)
        first_col[0] = 1
        A[:,0] = first_col


        for i in range(1,n_eles,1):
            for j in range(1,n_eles):
                A[i][j] = ((-j)**i)/np.math.factorial(i) 

        kernel = np.linalg.inv(A).dot(B)
        return kernel[::-1]
    
    if scheme=='central':
        return torch.from_numpy(np.array(central(diff_order,acc_order,dim))).reshape((1,1,-1)).float()
    elif scheme=='left':
        return torch.from_numpy(np.array(left(diff_order,acc_order,dim))).reshape((1,1,-1)).float()
    elif scheme=='right':
        return torch.from_numpy(np.array(right(diff_order,acc_order,dim))).reshape((1,1,-1)).float()

    
    
def padInputTorch(X,diff_order,acc_order=2,dim=None):
    '''
        pads the reqruied amount in the start and the end takes the input (batch_size x n_channels(1) x X_dim
    '''
    if dim is None:
        dim = diff_order+acc_order-1
    def padStart(X,diff_order,acc_order):
        central_kernel = getKernelTorch(diff_order,acc_order,scheme='central')
        right_kernel = getKernelTorch(diff_order,acc_order,scheme='right')
        n_c = central_kernel.shape[-1]
        n_r = right_kernel.shape[-1]
        n_eles = diff_order+acc_order-1 #central kernel
        n_pad = int(np.floor(n_eles/2))
        index = int(np.floor(n_c/2))-1
        for i in range(n_pad):
            right_conv = F.conv1d(X[...,index:index+n_r],right_kernel)
            central_conv = F.conv1d(X[...,0:n_c-1],central_kernel[...,1:])
            x_prev = (right_conv-central_conv)*1.0/central_kernel[...,0]
            X = torch.cat((x_prev,X),axis=-1)
        return X

    def padEnd(X,diff_order,acc_order):
        central_kernel = getKernelTorch(diff_order,acc_order,scheme='central')
        left_kernel = getKernelTorch(diff_order,acc_order,scheme='left')
        n_c = central_kernel.shape[-1]
        n_l = left_kernel.shape[-1]
        n_eles = diff_order+acc_order-1 #central kernel
        n_pad = int(np.floor(n_eles/2))
        index = X.shape[-1] - (int(np.floor(n_c/2))-1)
        
        for i in range(n_pad):       
            left_conv = F.conv1d(X[...,index-n_l:index],left_kernel)
            central_conv = F.conv1d(X[...,-(n_c-1):],central_kernel[...,:-1])
            x_next = (left_conv-central_conv)*1.0/central_kernel[...,-1]
            X = torch.cat((X,x_next),axis=-1)
            index += 1
        return X
    X = X.float()
    X = padStart(X,diff_order,acc_order)
    X = padEnd(X,diff_order,acc_order)
    
    actual_dim = diff_order+acc_order-1
    if actual_dim%2 == 0:
        actual_dim += 1
    npad = (int((dim-actual_dim)/2),int((dim-actual_dim)/2))
    X = F.pad(X,npad)
#     X[torch.isnan(X)] = 0

    return X


## data


class DataModel:
    def __init__(self,data_name,Nt,Nx,dt,dx,**kw):
        self.data_name = data_name
        self.Nt = Nt
        self.Nx = Nx
        self.dt = dt
        self.dx = dx
    
    @property
    def Lx(self):
        return self.dx*self.Nx
    
    @property
    def Lt(self):
        return self.dt*self.Nt
    
    def data(self):
        '''returns data
        '''
        raise NotImplementedError
        
    def _getMeshPoints(self):
        x = np.linspace(0,self.Lx,self.Nx)
        t = np.linspace(0,self.Lt,self.Nt)
        return x,t

            
    def visualize(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(self.x,self.t)        
        surf = ax.plot_surface(T, X, self.u_viz)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.show()
        
    


class TimeStepper:
    def __init__(self,dx,dt,time_scheme='rk4',acc_order=2,**kw):
        self.dx = dx
        self.dt = dt
        self.acc_order = acc_order
        self.time_scheme = time_scheme
    def _step_(self, ut):
        dt = self.dt
        if self.time_scheme == 'rk4': # classical 4-stage 4th-order Runge–Kutta method
            K1 = self.RightHandItems(ut)
            K2 = self.RightHandItems(ut+dt/2*K1)
            K3 = self.RightHandItems(ut+dt/2*K2)
            K4 = self.RightHandItems(ut+dt*K3)
            rhi = dt/6*(K1+2*K2+2*K3+K4)
        elif self.time_scheme == 'rk2': # 2-stage 2nd-order Runge–Kutta method
            K1 = self.RightHandItems(ut)
            K2 = self.RightHandItems(ut+dt*K1)
            rhi = dt/2*(K1+K2)
        else: # forward Euler
            rhi = dt*self.RightHandItems(ut)
        return rhi
    
    def step(self, ut):
        ut_1 = ut+self._step_(ut)
        return ut_1

    
    #takes the function at time t 
    def RightHandItems(self, ut):
            raise NotImplementedError


    

    
class DiffusionData(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,Nx,dt,dx,diff_coeff,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 2*Nx
        super(DiffusionData,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.diff_coeff = diff_coeff
        self.acc_order = acc_order
        self.batch_size = batch_size
#         self.ts = TimeStepper(self.dx,self.dt,time_scheme=time_scheme)
        self._data = self._generateData()

    
    def _generateData(self):
        start = int(self.Nx/4)
        end = 3*int(self.Nx/4)
        self.x,self.t =  self._getMeshPoints()
        u0 = self._IGaussian() #batch_size x 1 x X_dim
        u0[...,0] = u0[...,-1] = 0
        u0 = (u0-u0.min(axis=-1)[:,np.newaxis,:])/((u0.max(axis=-1)-u0.min(axis=-1))[:,np.newaxis,:])
        u = []   
        ut = torch.from_numpy(u0)
        u.append(torch.from_numpy(u0[...,start:end]))
        for i in range(1,self.Nt):
            u_next = self.step(ut)
            #boundary condtion
            u_next[...,0] = 0
            u_next[...,-1] = 0
            ut = u_next
            u.append(u_next[...,start:end])

        return u
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            F_c = self.diff_coeff/self.dx**2
            kernel = getKernelTorch(2,self.acc_order)
            ut_padded = padInputTorch(ut,2,self.acc_order)
            uxx_t = F.conv1d(ut_padded,kernel)
            return F_c*uxx_t

    
    
class BurgersEqn(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,Nx,dt,dx,viscosity,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        super(BurgersEqn,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
        self.batch_size = batch_size
        self._data = self._generateData()   
    
        
    def _generateData(self):
        self.x,self.t =  self._getMeshPoints()
        u0 = self._IGaussian() #batch_size x 1 x X_dim
        u0[...,0] = u0[...,-1] = 0
        u0 = (u0-u0.min(axis=-1)[:,np.newaxis,:])/((u0.max(axis=-1)-u0.min(axis=-1))[:,np.newaxis,:])
        u = []   
        ut = torch.from_numpy(u0)
        u.append(torch.from_numpy(u0))
        for i in range(1,self.Nt):
            u_temp = np.zeros((self.batch_size,1,self.Nx))
            u.append(u_temp)
            u[i] = self.step(ut)
            #boundary condtion
            u[i][...,0] = 0
            u[i][...,-1] = 0
            ut = u[i]
        return u
        
    def _IGaussian(self):
        sigma = np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            #1st order
            kernel_or1 = getKernelTorch(1,self.acc_order)
            ut_padded_or1 = padInputTorch(ut,1,self.acc_order)
            ux_t = F.conv1d(ut_padded_or1,kernel_or1)/self.dx     
            #2nd order 
            kernel_or2 = getKernelTorch(2,self.acc_order)
            ut_padded_or2 = padInputTorch(ut,2,self.acc_order)
            uxx_t = F.conv1d(ut_padded_or2,kernel_or2)/self.dx**2
            
            return self.viscosity*uxx_t - ut*ux_t
        
        
### Trigno


class DiffusionDataTrign(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,dt,dx,diff_coeff,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 128
        super(DiffusionDataTrign,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.diff_coeff = diff_coeff
        self.acc_order = acc_order
        self.batch_size = batch_size
#         self.ts = TimeStepper(self.dx,self.dt,time_scheme=time_scheme)
        self._data = self._generateData()

    
    def _generateData(self):
        start = 48
        end = 80
        self.x,self.t =  self._getMeshPoints()
        u0 = self._ITrigno() #batch_size x 1 x X_dim
        u0[...,0] = u0[...,-1] = 0
        u = []   
        u_whole = []
        ut = torch.from_numpy(u0)
        u_whole.append(torch.from_numpy(u0))
        u.append(torch.from_numpy(u0[...,start:end]))
        for i in range(1,self.Nt):
            u_next = self.step(ut)
            #boundary condtion
            u_next[...,0] = 0
            u_next[...,-1] = 0
            ut = u_next
            u.append(u_next[...,start:end])
            u_whole.append(u_next)
            
        self.u_all = u_whole
        return u
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    
    def _ITrigno(self):
        batch_size = self.batch_size
        w0 = 0.0
        c = np.random.uniform(-2,2,(batch_size,1))
        for k in range(-4,5):
            mew_k = np.random.normal(0,1,(batch_size,1))
            lambda_k = np.random.normal(0,1,(batch_size,1))
            w0 += mew_k*np.sin(k*self.x) + lambda_k*np.cos(k*self.x)

        I = (2*w0)/np.abs(w0).max(axis=-1)[:,np.newaxis]+c
        return I[:,np.newaxis,:]
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            F_c = self.diff_coeff/self.dx**2
            kernel = getKernelTorch(2,self.acc_order)
            ut_padded = padInputTorch(ut,2,self.acc_order)
            uxx_t = F.conv1d(ut_padded,kernel)
            return F_c*uxx_t
    
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        disp_arr = self.u_all
        x = self.x
        if subset:
            disp_arr = self._data
            x = self.x[48:80]
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b]).reshape(-1))
        u = np.array(u)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.show()
        
        return u
    
    
    
    
class BurgersEqnTrign(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,dt,dx,viscosity,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 128
        super(BurgersEqnTrign,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
        self.batch_size = batch_size
        self._data = self._generateData()   
    
        
    def _generateData(self):
        start = 48
        end = 80
        self.x,self.t =  self._getMeshPoints()
        u0 = self._ITrigno() #batch_size x 1 x X_dim
        u0[...,0] = u0[...,-1] = 0
        u = []   
        u_whole = []
        ut = torch.from_numpy(u0)
        u_whole.append(torch.from_numpy(u0))
        u.append(torch.from_numpy(u0[...,start:end]))
        for i in range(1,self.Nt):
            u_next = self.step(ut)
            #boundary condtion
            u_next[...,0] = 0
            u_next[...,-1] = 0
            ut = u_next
            u.append(u_next[...,start:end])
            u_whole.append(u_next)
            
        self.u_all = u_whole
        return u
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def _ITrigno(self):
        batch_size = self.batch_size
        w0 = 0.0
        c = np.random.uniform(-2,2,(batch_size,1))
        for k in range(-4,5):
            mew_k = np.random.normal(0,1,(batch_size,1))
            lambda_k = np.random.normal(0,1,(batch_size,1))
            w0 += mew_k*np.sin(k*self.x) + lambda_k*np.cos(k*self.x)

        I = (2*w0)/np.abs(w0).max(axis=-1)[:,np.newaxis]+c
        return I[:,np.newaxis,:]
    
    def data(self,step_num):
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, ut):
            ## should calculate the rhs a*d2u/dx2
            #1st order
            kernel_or1 = getKernelTorch(1,self.acc_order)
            ut_padded_or1 = padInputTorch(ut,1,self.acc_order)
            ux_t = F.conv1d(ut_padded_or1,kernel_or1)/self.dx     
            #2nd order 
            kernel_or2 = getKernelTorch(2,self.acc_order)
            ut_padded_or2 = padInputTorch(ut,2,self.acc_order)
            uxx_t = F.conv1d(ut_padded_or2,kernel_or2)/self.dx**2
            
            return self.viscosity*uxx_t - ut*ux_t

        
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        disp_arr = self.u_all
        x = self.x
        if subset:
            disp_arr = self._data
            x = self.x[48:80]
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b]).reshape(-1))
        u = np.array(u)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.show()
        
        return u

    
##generates 128x128 and takes 32x32
## this is to test 2 channels
class CrossAdvection(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,dt,dx,viscosity,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 128
        super(CrossAdvection,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
        self.batch_size = batch_size
        self._data = self._generateData()   
    
        
    def _generateData(self):
        start = 48
        end = 80
        self.x,self.t =  self._getMeshPoints()
        u0 = self._ITrigno() #batch_size x 1 x X_dim # for u dimension
        v0 = self._ITrigno() #batch_size x 1 x X_dim # for v dimension
        
        u0[...,0] = u0[...,-1] = 0
        v0[...,0] = v0[...,-1] = 0
        
        #capital U denotes the u and v both
        U0 = np.concatenate((u0,v0),axis=1) ## batch_size x 2 x X_dim

        U = []
        U_whole = []
        
        Ut = torch.from_numpy(U0)
        U_whole.append(Ut)       
        U.append(torch.from_numpy(U0[...,start:end]))

        for i in range(1,self.Nt):
            U_next = self.step(Ut)
            #boundary condtion
            U_next[...,0] = 0
            U_next[...,-1] = 0
            Ut = U_next
            U.append(U_next[...,start:end])
            U_whole.append(U_next)
            
        self.U_all = U_whole
        return U
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def _ITrigno(self):
        batch_size = self.batch_size
        w0 = 0.0
        c = np.random.uniform(-2,2,(batch_size,1))
        for k in range(-4,5):
            mew_k = np.random.normal(0,1,(batch_size,1))
            lambda_k = np.random.normal(0,1,(batch_size,1))
            w0 += mew_k*np.sin(k*self.x) + lambda_k*np.cos(k*self.x)

        I = (2*w0)/np.abs(w0).max(axis=-1)[:,np.newaxis]+c
        return I[:,np.newaxis,:]
    
    def data(self,step_num):
        '''
            returns of shape batch_size X 2 X X_dim
        '''
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, Ut):
            #1st order
            U_split = Ut.split(1,dim=1)
            U_next = []
            
            kernel_or1 = getKernelTorch(1,self.acc_order)
            kernel_or2 = getKernelTorch(2,self.acc_order)
            for i in range(2):
                Ut_padded_or1 = padInputTorch(U_split[i],1,self.acc_order)
                Ux_t = F.conv1d(Ut_padded_or1,kernel_or1)/self.dx     
                #2nd order 
                Ut_padded_or2 = padInputTorch(U_split[i],2,self.acc_order)
                Uxx_t = F.conv1d(Ut_padded_or2,kernel_or2)/self.dx**2
                
                U_next.append(self.viscosity*Uxx_t - U_split[(i+1)%2]*Ux_t)
                
            U_next = torch.cat(U_next,dim=1)
            return U_next

        
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        v = []
        disp_arr = self.U_all
        x = self.x
        if subset:
            disp_arr = self._data
            x = self.x[48:80]
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b][0]).reshape(-1))
            v.append(np.array(disp_arr[i][b][1]).reshape(-1))
            
        u = np.array(u)
        v = np.array(v)
        
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("u")
        plt.show()
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, v)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("v")
        plt.show()
        

    
    

class _MK(torch.nn.Module):
    def __init__(self, dim):
        '''
        '''
        super(_MK, self).__init__()
        self._dim = dim
        M = np.zeros((dim,dim))
        for i in range(dim):
            M[i] = ((np.arange(dim)-(dim-1)//2)**i)/np.math.factorial(i)
        invM = np.linalg.inv(M)
        self.register_buffer('_M', torch.from_numpy(M))
        self.register_buffer('_invM', torch.from_numpy(invM))

    @property
    def M(self):
        return self._buffers['_M']
    @property
    def invM(self):
        return self._buffers['_invM']

    
    def dim(self):
        return self._dim

    def forward(self):
        pass


    
    
class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        dim (int): kernel dim
    """
    def __init__(self, dim):
        super(M2K, self).__init__(dim)
    def forward(self, m):
        """
        m (Tensor): torch.size=(1,1,dim),
        returns kernel of the same shape
        """
        x = torch.tensordot(self.invM, m, dims=[[1],[-1]])
        x = x.permute([2,1,0])
        return x
    
    
class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        dim (int): kernel dim
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=(1,1,dim),
        returns moment of the same shape
        """
        x = torch.tensordot(self.M, k, dims=[[1],[-1]])
        x = x.permute([2,1,0])
        return x
    

class EnergyExchange(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,dt,dx,viscosity,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 128
        super(EnergyExchange,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
        self.batch_size = batch_size
        self._data = self._generateData()   
    
        
    def _generateData(self):
        start = 48
        end = 80
        self.x,self.t =  self._getMeshPoints()
        u0 = self._ITrigno() #batch_size x 1 x X_dim # for u dimension
        v0 = self._ITrigno() #batch_size x 1 x X_dim # for v dimension
        
        u0[...,0] = u0[...,-1] = 0
        v0[...,0] = v0[...,-1] = 0
        
        #capital U denotes the u and v both
        U0 = np.concatenate((u0,v0),axis=1) ## batch_size x 2 x X_dim
#         print("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}")
#         print(u0)
#         print("+++++++++++++++++++++++++++")
#         print(v0)
        
#         print("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}")

        U = []
        U_whole = []
        
        Ut = torch.from_numpy(U0)
        U_whole.append(Ut)       
        U.append(torch.from_numpy(U0[...,start:end]))

        for i in range(1,self.Nt):
            U_next = self.step(Ut)
            #boundary condtion
            
            U_next[...,0] = 0
            U_next[...,-1] = 0
            Ut = U_next
            U.append(U_next[...,start:end])
            U_whole.append(U_next)
            
        self.U_all = U_whole
        return U
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def _ITrigno(self):
        batch_size = self.batch_size
        w0 = 0.0
        c = np.random.uniform(-2,2,(batch_size,1))
        for k in range(-4,5):
            mew_k = np.random.normal(0,1,(batch_size,1))
            lambda_k = np.random.normal(0,1,(batch_size,1))
            w0 += mew_k*np.sin(k*self.x) + lambda_k*np.cos(k*self.x)

        I = (2*w0)/np.abs(w0).max(axis=-1)[:,np.newaxis]+c
        return I[:,np.newaxis,:]
    
    def data(self,step_num):
        '''
            returns of shape batch_size X 2 X X_dim
        '''
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, Ut):
            #1st order
            U_split = Ut.split(1,dim=1)
            U_next = []
#             if True in torch.isnan(Ut):
#                 print(Ut)
#                 raise "khatam"
            kernel_or1 = getKernelTorch(1,self.acc_order)
            kernel_or2 = getKernelTorch(2,self.acc_order)
            
            #du/dx
            Ut_padded_or1 = padInputTorch(U_split[0],1,self.acc_order)
            
            
            Ux_t = F.conv1d(Ut_padded_or1,kernel_or1)/self.dx  
            #dv/dx
            Vt_padded_or1 = padInputTorch(U_split[1],1,self.acc_order)
            Vx_t = F.conv1d(Vt_padded_or1,kernel_or1)/self.dx  
            
            for i in range(2):
                #2nd order 
                Ut_padded_or2 = padInputTorch(U_split[i],2,self.acc_order)
                Uxx_t = F.conv1d(Ut_padded_or2,kernel_or2)/self.dx**2                
                
                U_next.append(self.viscosity*Uxx_t + ((-1)**(i+1))*U_split[0]*U_split[1])
                
            U_next = torch.cat(U_next,dim=1)
            if True in torch.isinf(U_next):
                print("*********************************************")
                print(Ut)
                print("*********************************************")
                print(U_next)
                raise "dsadasd"
            return U_next

        
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        v = []
        disp_arr = self.U_all
        x = self.x
        if subset:
            disp_arr = self._data
            x = self.x[48:80]
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b][0]).reshape(-1))
            v.append(np.array(disp_arr[i][b][1]).reshape(-1))
            
        u = np.array(u)
        v = np.array(v)
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("u")
        plt.show()
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, v)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("v")
        plt.show()
        
        
        
##generates 128x128 and takes 32x32
## this is to test 2 channels
class EnergyExchangeAdvan(DataModel,TimeStepper):
    def __init__(self,data_name,Nt,dt,dx,viscosity,batch_size,acc_order=2,time_scheme='rk4'):
        '''
            currently only batch size of 1 is supported
        '''
        Nx = 128
        super(EnergyExchangeAdvan,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx,time_scheme=time_scheme\
                                          ,acc_order=acc_order)
        self.time_scheme = time_scheme
        self.viscosity = viscosity
        self.acc_order = acc_order
        self.batch_size = batch_size
        self._data = self._generateData()   
    
        
    def _generateData(self):
        start = 48
        end = 80
        self.x,self.t =  self._getMeshPoints()
        u0 = self._ITrigno() #batch_size x 1 x X_dim # for u dimension
        v0 = self._ITrigno() #batch_size x 1 x X_dim # for v dimension
        
        u0[...,0] = u0[...,-1] = 0
        v0[...,0] = v0[...,-1] = 0
        
        #capital U denotes the u and v both
        U0 = np.concatenate((u0,v0),axis=1) ## batch_size x 2 x X_dim
#         print("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}")
#         print(u0)
#         print("+++++++++++++++++++++++++++")
#         print(v0)
        
#         print("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}")

        U = []
        U_whole = []
        
        Ut = torch.from_numpy(U0)
        U_whole.append(Ut)       
        U.append(torch.from_numpy(U0[...,start:end]))

        for i in range(1,self.Nt):
            U_next = self.step(Ut)
            #boundary condtion
            
            U_next[...,0] = 0
            U_next[...,-1] = 0
            Ut = U_next
            U.append(U_next[...,start:end])
            U_whole.append(U_next)
            
        self.U_all = U_whole
        return U
        
    def _IGaussian(self):
        sigma = 2*np.random.uniform(1,2,self.batch_size).reshape(-1,1)
        scale = 2*np.random.uniform(0,1,self.batch_size).reshape(-1,1)
        return (scale*np.exp(-0.5*((self.x-self.Lx/2.0)**2)/sigma**2)).reshape((self.batch_size,1,-1))
    
    def _ITrigno(self):
        batch_size = self.batch_size
        w0 = 0.0
        c = np.random.uniform(-2,2,(batch_size,1))
        for k in range(-4,5):
            mew_k = np.random.normal(0,1,(batch_size,1))
            lambda_k = np.random.normal(0,1,(batch_size,1))
            w0 += mew_k*np.sin(k*self.x) + lambda_k*np.cos(k*self.x)

        I = (2*w0)/np.abs(w0).max(axis=-1)[:,np.newaxis]+c
        return I[:,np.newaxis,:]
    
    def data(self,step_num):
        '''
            returns of shape batch_size X 2 X X_dim
        '''
        return self._data[:step_num]
    
    #takes the function at time t 
    def RightHandItems(self, Ut):
            #1st order
            U_split = Ut.split(1,dim=1)
            U_next = []
#             if True in torch.isnan(Ut):
#                 print(Ut)
#                 raise "khatam"
            kernel_or1 = getKernelTorch(1,self.acc_order)
            kernel_or2 = getKernelTorch(2,self.acc_order)
            
            #du/dx
            Ut_padded_or1 = padInputTorch(U_split[0],1,self.acc_order)
            
            
            Ux_t = F.conv1d(Ut_padded_or1,kernel_or1)/self.dx  
            #dv/dx
            Vt_padded_or1 = padInputTorch(U_split[1],1,self.acc_order)
            Vx_t = F.conv1d(Vt_padded_or1,kernel_or1)/self.dx  
            
            for i in range(2):
                #2nd order 
                Ut_padded_or2 = padInputTorch(U_split[i],2,self.acc_order)
                Uxx_t = F.conv1d(Ut_padded_or2,kernel_or2)/self.dx**2                
                
                U_next.append(self.viscosity*Uxx_t + ((-1)**(i+1))*Ux_t*Vx_t)
                
            U_next = torch.cat(U_next,dim=1)
            if True in torch.isinf(U_next):
                print("*********************************************")
                print(Ut)
                print("*********************************************")
                print(U_next)
                raise "dsadasd"
            return U_next

        
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        v = []
        disp_arr = self.U_all
        x = self.x
        if subset:
            disp_arr = self._data
            x = self.x[48:80]
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b][0]).reshape(-1))
            v.append(np.array(disp_arr[i][b][1]).reshape(-1))
            
        u = np.array(u)
        v = np.array(v)
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("u")
        plt.show()
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,self.t)        
        surf = ax.plot_surface(T, X, v)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title("v")
        plt.show()
        

        
DT = 1800.
L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_F = L_I
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5
RHO_L = 1e3








## keep a num_channel_recoverable (for tacking LHFLX,SHFLX variables)
## keep a num_channel_recoverable (for tacking LHFLX,SHFLX variables)
## keep a num_channel_recoverable (for tacking LHFLX,SHFLX variables)
class ClimateData(DataModel):
    def __init__(self,data_name,Nx,Nt,dt,dx,batch_size,channel_names,channel_const_dict,\
                 non_eqn_channel, data_file,region=0):
        '''
            currently only batch size of 1 is supported
            Nx is the interpolated dimension
            Nt max can be 240 (this will take the middle 240 from 7 days i.e skip the 1st day and the last day)
            dt is fixed which is 1800 #can't change 
            dx (the lev tilde will be interpolated from 0 to Nx*dx)
            batch_size: as of now max is 48 (1 year data)
            region: It is the lat_lon clustering to load
            data_dir: location of the xarray data
            channel_names: list of channels (variables) ['TAP','QAP']
            channel_const_dict: constant multiplier dict {'TAP':C_P, 'QAP':L_V}
            non_eqn_channel: list of channels(variables) without recoverable eqn ['LHFLX','SHFLX'] (time x batch)
                basically not dependent on lev
        '''
        super(ClimateData,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx)
        self.batch_size = batch_size
        self.region = region
        self.ds = xr.open_dataset(data_file)
        self.channel_names = channel_names
        self.channel_const_dict = channel_const_dict
        self.diff_dict = {"TAP":"DTV","QAP":"VD01","TBP":"TPHYSTND","QBP":"PHQ","TCRM":"TCDTAdiab","QCRM":"QCDTAdiab"}
        self.non_eqn_channel = non_eqn_channel
        self.n_eqn = len(channel_names)
        self.n_non_eqn = len(non_eqn_channel)
        self._data,self._future_dict = self._generateData() ## returns data to be fed #time x batch x lev,
        ##future dict is the 
    
        
    def _generateData(self):
        start = (336-self.Nt)//2
        end = start+self.Nt
        
        
        ##need to interpolate all variables.
        #get t2tns for getting intermediate lev
        TBP = self.ds['TBP'][start:end].values 
        t2tns = self._getT2Tns(TBP)
        lev = self.ds.lev.values
        
        U = []
        eqn_channels_vars = []
        non_eqn_channels_vars = []
        diff_dict_key_name = list(self.diff_dict.values())
        diff_values_vars = []
        diff_dict_keys = list(self.diff_dict.keys())

        #shape the eqn vars
        for var in self.channel_names:
            v = self.ds[var][start:end,:self.batch_size].values[:,:,np.newaxis,:]
            eqn_channels_vars.append(self.channel_const_dict[var]*v)
        
        for i,var in enumerate(diff_dict_key_name):
            v = self.ds[var][start:end,:self.batch_size].values[:,:,np.newaxis,:]
            if 'T' in diff_dict_keys[i]:
                var_n = 'TAP'
            else:
                var_n = 'QAP'
            diff_values_vars.append(self.channel_const_dict[var_n]*v)

        #shape the non eqn vas
        for var in self.non_eqn_channel:
            v = self.ds[var][start:end,:self.batch_size].values[:,:,np.newaxis]
            v = np.tile(v,[1,1,self.Nx])
            v = v[:,:,np.newaxis,:]
            v = v*G/P0
            non_eqn_channels_vars.append(v)
        
        # interpolate the eqn variables
        self.lev_tilde_after =  np.linspace(0,self.Lx,num=self.Nx)
        imin = np.argmin(t2tns[...,6:],axis=-1) + 6
        lev_stacked = np.tile(lev[np.newaxis,np.newaxis,:],[self.Nt,self.batch_size,1])
        lev_roof = lev[imin] #time x batch_size
        #timexbatchx30
        lev_tilde = (lev_stacked[...,-1][...,np.newaxis]-lev_stacked[:])/(lev_stacked[...,-1][...,np.newaxis]-lev_roof[:,:,np.newaxis])
        
        eqn_channels_vars_interp = []
        diff_values_vars_interp = []
        
        for i,v in enumerate(eqn_channels_vars+diff_values_vars):
            batch_size = v.shape[1]
            v_interp = np.zeros(v.shape[:-1]+(self.Nx,))
            
            for t in range(self.Nt):
                for b in range(batch_size):
                    interp = np.interp(self.lev_tilde_after,np.flip(lev_tilde[t][b]),np.flip(v[t][b][0]))
                    v_interp[t][b][0] = interp
        
            if i<self.n_eqn:
                eqn_channels_vars_interp.append(v_interp) 
#                 eqn_channels_vars_interp.append(v_interp/v_interp.max(axis=-1)[...,np.newaxis])        
                
            else:
                diff_values_vars_interp.append(v_interp)
#                 diff_values_vars_interp.append(v_interp/v_interp.max(axis=-1)[...,np.newaxis])
                
                
        
        U = np.concatenate(eqn_channels_vars_interp,axis=2)
        dict_ = {"AP":[],"BP":[],"CRM":[]}
        dict_keys = list(dict_.keys())
        for i in range(3):
            dict_[dict_keys[i]] = torch.from_numpy(np.concatenate(diff_values_vars_interp[2*i:2*i+2],axis=2)).type(torch.DoubleTensor)
        self._U_noneqn = np.concatenate(non_eqn_channels_vars,axis=2)
        return U,dict_
        
    
    def _getT2Tns(self,tbp):
        #time,batch,lev
        t2tns = tbp-np.tile(np.expand_dims(tbp[...,-1],axis=-1),[1,1,30])
        return t2tns
    
    def data(self,step_num):
        '''
            returns of shape step_num x batch_size X 2 X X_dim
        '''
        return torch.from_numpy(self._data[:step_num]),torch.from_numpy(self._U_noneqn[:step_num]).type(torch.DoubleTensor)
    
        
    def visualize(self,b,subset=True):
        #displays ith batch plot
        ##time x batch_size x 1 x x_dim 
        u = []
        v = []
        x,t = self._getMeshPoints()
        disp_arr = self._data
            
        for i in range(len(self._data)):
            u.append(np.array(disp_arr[i][b][0]).reshape(-1))
            v.append(np.array(disp_arr[i][b][1]).reshape(-1))
            
        u = np.array(u)
        v = np.array(v)
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,t)        
        surf = ax.plot_surface(T, X, u)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title(self.channel_names[0])
        plt.show()
        
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        X,T = np.meshgrid(x,t)        
        surf = ax.plot_surface(T, X, v)
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.title(self.channel_names[1])
        plt.show()
        
        

class SymNet(torch.nn.Module):
    def __init__(self, n_hidden, n_deriv_channel, deriv_channel_names=None, normalization_weight=None):
        '''
        Input:
            n_channel = Number of derivatives using (u,ux,vx,vxx)
        '''
        super(SymNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_deriv_channel = n_deriv_channel
        if deriv_channel_names is None:
            deriv_channel_names = list('u_'+str(i) for i in range(self.n_deriv_channel))
        self.deriv_channel_names = deriv_channel_names
        layers = []
        for k in range(n_hidden):
            module = torch.nn.Linear(n_deriv_channel+k,2)
            self.add_module('layer'+str(k), module)
            layers.append(self.__getattr__('layer'+str(k)))
        module = torch.nn.Linear(n_deriv_channel+n_hidden, 1)
        self.add_module('layer_final', module)
        layers.append(self.__getattr__('layer_final'))
        self.layers = tuple(layers)
        
        
    def forward(self, inputs):
        '''
            inputs tensor be of shape (batch_size x X_dim x n_derivatives)
            output shape batch_size x X_dim
        '''
        outputs = inputs.type(torch.FloatTensor)
        for k in range(self.n_hidden):
            
            o = self.layers[k](outputs)
            outputs = torch.cat([outputs,o[...,:1]*o[...,1:]], dim=-1)
        outputs = self.layers[-1](outputs)
        return outputs[...,0]
    
    
    
    def _cast2symbol(self,layer):
        weight,bias = layer.weight.data.cpu().numpy(), \
                    layer.bias.data.cpu().numpy()
        weight,bias = sympy.Matrix(weight),sympy.Matrix(bias)
        return weight,bias

    def _sympychop(self,o, calprec):
        for i in range(o.shape[0]):
            cdict = o[i].expand().as_coefficients_dict()  
            o_i = 0
            for k,v in cdict.items():
                if abs(v)>0.1**calprec:
                    o_i = o_i+k*v
            o[i] = o_i
        return o

    def getEquation(self,calprec=6):
        ## assume symnet model

        deriv_channels = sympy.symbols(self.deriv_channel_names)
        deriv_channels = sympy.Matrix([deriv_channels,])
        for i in range(self.n_hidden):
            weight,bias = self._cast2symbol(self.layers[i])
            o = weight*deriv_channels.transpose()+bias
            o = self._sympychop(o, calprec) #ignores very low params terms
            deriv_channels = list(deriv_channels)+[o[0]*o[1],]
            deriv_channels = sympy.Matrix([deriv_channels,])

        weight,bias = self._cast2symbol(self.layers[-1])
        o = (weight*deriv_channels.transpose()+bias)
        o = self._sympychop(o,calprec)

        return o[0]

    
class FD1D(torch.nn.Module):
    '''
        Finite Differences scheme for 1D dependency
        acc_order list of acc_order for diff_order
        kernel size should be greater than or equal to the max_diff_order (otherwise error will be thrown)
        diff_order starts from 0,1,... 
        constarint: Moment matrix constraint "free" or "moment"
        Basically this class initializes one kernel of the specified parameters
    '''
    def __init__(self,dx, kernel_size, diff_order,acc_order,constraint='free'):
        super(FD1D, self).__init__()
        self.dx = dx
        self.kernel_size = kernel_size
        self.diff_order = diff_order
        self.acc_order = acc_order
        self.constraint = constraint
        ## will only be used in case of contraint = 'free'
        if constraint=='free':
            self._kernel = (getKernelTorch(diff_order,acc_order+1,dim=kernel_size,scheme='central')/(dx**diff_order)).type(torch.DoubleTensor)
            
        ##Moment to kernel and vice versa
        if constraint=='moment':
            self.m2k = M2K(kernel_size)
            self.k2m = K2M(kernel_size)
            #define moment matrix
            moment = torch.DoubleTensor(kernel_size).zero_()
            moment[diff_order] = 1
            moment = moment.reshape(1,1,-1)
            self.moment = torch.nn.Parameter(moment) ## now weights will be updated on this
            ##create a mask for gradeint hook
            self.gradient_mask = self._getGradientMask()
            ##register hook to the moment matrix
            self.moment.register_hook(lambda grad: grad.mul_(self.gradient_mask))
        
        
    @property
    def kernel(self):
        if self.constraint == 'moment':
            kernel = self.m2k(self.moment)/(self.dx**self.diff_order)
        else:
            kernel = self._kernel
            
            
        return kernel
    
    def _getGradientMask(self):
        gradient_mask = torch.ones(self.kernel_size,dtype=torch.double)
        order_bank = np.arange(self.kernel_size)
        for j in range(self.diff_order+self.acc_order):
            gradient_mask[order_bank[j]] = 0
        gradient_mask = gradient_mask.reshape(1,1,-1)
        return gradient_mask
        
    def forward(self,inputs):
        '''
            Process:
            Need to pad the input and then appy conv1D
            input shape can be batch_size x n_channels x x_dim
        '''
        inp_padded = padInputTorch(inputs,self.diff_order,self.acc_order+1,dim=self.kernel_size) #batch_size x n_channels x (x_dim+padded)
        conv = F.conv1d(inp_padded.type(torch.DoubleTensor),self.kernel)
        return conv
        
        
def symnetRegularizeLoss(model):
    loss = 0
    s = 1e-2
    for p in model.symNetParams():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss

def momentRegularizeLoss(model):
    loss = 0
    s = 1e-2
    for p in model.diffParams():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()

    return loss