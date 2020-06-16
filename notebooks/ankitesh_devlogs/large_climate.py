#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import torch
import sympy
import torch.nn.functional as F
from backend.utils import *
import pickle
import configparser
import yaml
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from backend.lbfgsnew import *


# In[6]:


## keep a num_channel_recoverable (for tacking LHFLX,SHFLX variables)
class ClimateData(DataModel):
    def __init__(self,data_name,Nx,Nt,dt,dx,batch_size,channel_names,                 non_eqn_depen_channel, non_eqn_channel, data_file,scaling=1,total_points=-1):
        '''
            This supports the batch option (handy for large datasets)
        '''
        
        self.scaling = scaling
        dx = scaling*dx
        super(ClimateData,self).__init__(data_name=data_name,Nt=Nt, Nx=Nx, dt=dt, dx=dx)
        self.batch_size = batch_size
        self.ds = xr.open_dataset(data_file)
        if total_points!=-1:
            self.ds = self.ds.sel({"batch_size":slice(48*400,48*400+total_points)})
         
        self.channel_names = channel_names
        self.non_eqn_depen_channel = non_eqn_depen_channel
        self.n_non_eqn_depen = len(non_eqn_depen_channel)
        self.diff_dict = {"TAP":"DTV","QAP":"VD01","TBP":"TPHYSTND","QBP":"PHQ","TCRM":"TCDTAdiab","QCRM":"QCDTAdiab"}
        self.non_eqn_channel = non_eqn_channel
        self.n_eqn = len(channel_names)
        self.n_non_eqn = len(non_eqn_channel)
        total = self.ds.batch_size.size
        self.n_batch = total//batch_size
        self.curr_batch = 0
        self.sub, self.div = self._getGlobalStats()
        
    
    def _getGlobalStats(self):
        return self.ds.min(),self.ds.max()-self.ds.min()

    def _generateData(self):
        
        start = (336-self.Nt)//2
        end = start+self.Nt
        
        lev = self.scaling*self.ds.lev.values
        
        U = []
        eqn_channels_vars = []
        non_eqn_depen_channel_vars = []
        non_eqn_channels_vars = []
        diff_dict_key_name = list(self.diff_dict.values())
        diff_values_vars = []
        diff_dict_keys = list(self.diff_dict.keys())

        #shape the eqn vars
        for var in self.channel_names:
            v = self.ds[var][start:end,self.batch_start:self.batch_end].values[:,:,np.newaxis,:]
            v = (v-float(self.sub[var]))/float(self.div[var])
            eqn_channels_vars.append(v)
            
        for var in self.non_eqn_depen_channel:
            v = self.ds[var][start:end,self.batch_start:self.batch_end].values[:,:,np.newaxis,:]
            v = (v-float(self.sub[var]))/float(self.div[var])
            non_eqn_depen_channel_vars.append(v)
        
        for i,var in enumerate(diff_dict_key_name):
            v = self.ds[var][start:end,self.batch_start:self.batch_end].values[:,:,np.newaxis,:]
            if float(self.div[var]) !=0:
                v = (v-float(self.sub[var]))/float(self.div[var])
            diff_values_vars.append(v)
            
        n_diff_values_vars = len(diff_values_vars)
        #shape the non eqn vas
        for var in self.non_eqn_channel:
            v = self.ds[var][start:end,self.batch_start:self.batch_end].values[:,:,np.newaxis]
            v = np.tile(v,[1,1,self.Nx])
            v = v[:,:,np.newaxis,:]
            v = (v-float(self.sub[var]))/float(self.div[var])
            non_eqn_channels_vars.append(v)
        
        # interpolate the eqn variables
        self.lev_tilde_after =  np.linspace(0,self.Lx,num=self.Nx)

        eqn_channels_vars_interp = []
        non_eqn_depen_channel_vars_interp = []
        diff_values_vars_interp = []

        for i,v in enumerate(eqn_channels_vars+diff_values_vars+non_eqn_depen_channel_vars):
            batch_size = v.shape[1]
            v_interp = np.zeros(v.shape[:-1]+(self.Nx,))
            
            for t in range(self.Nt):
                for b in range(batch_size):
                    interp = np.interp(self.lev_tilde_after,lev,v[t][b][0])
                    v_interp[t][b][0] = interp
        
            if i<self.n_eqn:
                eqn_channels_vars_interp.append(v_interp) 
                
            elif i<self.n_eqn+n_diff_values_vars:
                diff_values_vars_interp.append(v_interp)
            else:
                non_eqn_depen_channel_vars_interp.append(v_interp)
                
                
        
        U = np.concatenate(eqn_channels_vars_interp,axis=2)
        dict_ = {"AP":[],"BP":[],"CRM":[]}
        dict_keys = list(dict_.keys())
        for i in range(3):
            dict_[dict_keys[i]] = torch.from_numpy(np.concatenate(diff_values_vars_interp[2*i:2*i+2],axis=2)).type(torch.DoubleTensor)

        _U_noneqn = np.concatenate(non_eqn_channels_vars,axis=2)
        _U_noneqn_dep = np.concatenate(non_eqn_depen_channel_vars_interp,axis=2)
        return U,_U_noneqn_dep,_U_noneqn,dict_
        

    def data(self,step_num):
        '''
            fetches the next batch
        '''
        self.batch_start = self.curr_batch*self.batch_size
        self.batch_end = self.batch_start + self.batch_size
        self.curr_batch += 1
        self._data,self._U_noneqn_dep,self._U_noneqn,self.dict_ = self._generateData()
        
        if self.curr_batch >= self.n_batch:
            self.curr_batch = 0
        return torch.from_numpy(self._data[:step_num]),torch.from_numpy(self._U_noneqn_dep[:step_num]),                torch.from_numpy(self._U_noneqn[:step_num]).type(torch.DoubleTensor),                self.dict_
                    
    
        
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
        
        


# In[7]:


'''
Modifying to incorporate untrainable/unlearnable equation channel with and without the lev derivative
'''

class PdeNet(torch.nn.Module):
    def __init__(self,dt, dx, kernel_size, max_diff_order, n_channel,channel_names,
                 n_non_eqn_channels,non_eqn_channel_names,dependent_channels,
                 acc_order=2,n_hidden=2,\
                constraint='free'):
        '''
        Input:
        '''
        super(PdeNet, self).__init__()
        self.dx = dx
        self.dt = dt
        self.kernel_size = kernel_size
        self.max_diff_order = max_diff_order
        self.n_channel = n_channel
        self.channel_names = channel_names
        self.n_non_eqn_channels = n_non_eqn_channels
        self.non_eqn_channel_names = non_eqn_channel_names
        self.dependent_channels = dependent_channels
        self.n_dependent_channels = len(dependent_channels)
        self.n_hidden = n_hidden
        self.constraint = constraint
                        
        if not np.iterable(acc_order):
            acc_order = [acc_order,]*(self.max_diff_order+1)
            
        self.acc_order = acc_order
        
        #conv operation
        for i in range(max_diff_order+1):
            kernel = FD1D(dx,kernel_size,i,acc_order[i],constraint=constraint)
            self.add_module('fd'+str(i), kernel) #finite difference of order
            
        #symnet 
        c = channel_names.split(',')
        derivative_channels = []
        for ch in c+self.dependent_channels:
            for k in range(max_diff_order+1):
                derivative_channels.append(ch+'_'+str(k))
                
        '''CHANGED: for climate''' 
        for ch in non_eqn_channel_names:
            derivative_channels.append(ch)
        
            
        self.derivative_channels = derivative_channels 
        all_symnets = []
        for k in range(self.n_channel):
            self.add_module("symnet_"+str(k),SymNet(n_hidden,len(derivative_channels), deriv_channel_names=derivative_channels))
            all_symnets.append(self.__getattr__('symnet_'+str(k)))
        self.all_symnets = all_symnets
    
    @property
    def fds(self):
        for i in range(self.max_diff_order+1):
            yield self.__getattr__('fd'+str(i))
                
    def multistep(self,inputs,non_eqn_depe,non_eqn_t,diff_values,step_num):
        #pass it throught the kernels then the symmnet to 
        '''
        Takes multistep through the whole PDE Net.
        '''
        u = inputs
        for i in range(step_num):
            uadd = self.RightHandItems(u,non_eqn_depe,non_eqn_t)#will take a dt step from u using the network 
            u = u + self.dt*(uadd+diff_values[0]+diff_values[1]) #only for QBP
        return u
    
    def symNetParams(self):
        params = []
        for symnet in self.all_symnets:
            params += list(symnet.parameters())
        return params
    
    def diffParams(self):
        params = []
        for fd in self.fds:
            params += list(fd.parameters())
        return params

    def RightHandItems(self,u,non_eqn_depe,non_eqn_t):
        
        #convolve the u with the derivative kernals to get the different derivatives 
        #batch_size x n_channels x X_dim
        derives = []
        u_split = u.split(1,dim=1)
        non_eqn_depe_split = non_eqn_depe.split(1,dim=1)
        for ch in range(self.n_channel):       
            for i in range(self.max_diff_order+1):
                fd_obj = self.__getattr__('fd'+str(i))
                deriv_channel_ch_order_i = fd_obj(u_split[ch])
                derives.append(deriv_channel_ch_order_i)   
        
        for ch in range(self.n_dependent_channels):
            for i in range(self.max_diff_order+1):
                fd_obj = self.__getattr__('fd'+str(i))
                deriv_channel_ch_order_i = fd_obj(non_eqn_depe_split[ch])
                derives.append(deriv_channel_ch_order_i)   
                
                
            
        U = torch.cat(derives, dim=1) #batch_size x n_derivatives x X_dim 
        U = torch.cat([U,non_eqn_t],dim=1)
        
        #symnet_output = (batch_size x X_dim x n_derivatives)
        u_outs = []
        for symnet in self.all_symnets:
            u_symnet = symnet(U.permute(0,2,1)) #batch_size x X_dim x n_derivatives
            u_out = u_symnet.unsqueeze_(1)
            u_outs.append(u_out)
        u_out = torch.cat(u_outs,axis=1)#only 1 channel as there will only be 1 symnet
        return u_out
        
        
    def forward(self,inputs,non_eqn_depe,non_eqn_t,diff_values,step_num):
        '''
            inputs of shape batch_size x n_channels x X_dim
            step_nums = number of dt blocks to calculate the inputs for
        '''
        return self.multistep(inputs,non_eqn_depe,non_eqn_t,diff_values,step_num)

    


# In[8]:


#global names are all the parameters
def modelLoss(model,u_obs,non_eqn_dep_obs,non_eqn_obs,diff_dict,config,block):
    '''
        Returns the loss value for so that it can be given to an optimizer
        Inputs:
            u_obs (batch_size x n_channels x X_dim)
            blocks is stepnum
    '''
    sparsity = config['sparsity']
    momentsparsity = config['momentsparsity']
    
    if block==0: #warmup
        sparsity = 0
        momentsparsity = 0
    step_num = block if block>=1 else 1
    dt = config['dt']
    data_loss = 0
    symnet_loss = symnetRegularizeLoss(model)
    moment_loss = momentRegularizeLoss(model)
    
    ut = u_obs[0]
    loss_mse = 0
    mse_loss = torch.nn.MSELoss()
    
    for steps in range(1,step_num+1):
        non_eqn_t = non_eqn_obs[steps-1]
        non_eqn_dep_t = non_eqn_dep_obs[steps-1]
        diff_values = [diff_dict['AP'][steps-1][:,-1:,:],diff_dict['CRM'][steps][:,-1:,:]] #for QBP
        ut_next_predicted = model(ut,non_eqn_dep_t,non_eqn_t,diff_values,step_num=1) #take one step from this point #only 1 channel(QBP)
        loss_mse_t = mse_loss(ut_next_predicted,u_obs[steps])
        loss_mse += loss_mse_t
        data_loss += (loss_mse_t/dt**2)/step_num
        ut = ut_next_predicted
        

    loss = data_loss+stepnum*sparsity*symnet_loss+stepnum*momentsparsity*moment_loss
    if torch.isnan(loss):
#         raise "Loss Nan"
        loss = (torch.ones(1,requires_grad=True)/torch.zeros(1)).to(loss)
    return loss,data_loss,symnet_loss,moment_loss,loss_mse


# In[9]:


##modify channel names and length
def setenv(config): #return model and datamodel
    model = PdeNet(config['dt'],config['dx']*config['scaling'],config['kernel_size'],config['max_diff_order']                   ,config['n_channels'],config['channel_names'],
                   config['n_non_eqn_channels'],config['non_eqn_channels'],
                   config['dependent_channels'],
                   config['acc_order'],config['n_hidden_layers'],config['constraint'])
    
    data_model =  ClimateData(config['dataname'],config['Nx'],config['Nt']
                              ,config['dt'],config['dx'],config['batch_size'],config['channel_vars'],\
                              config['dependent_channels'],
                              config['non_eqn_channels'],config['data_file'],config['scaling'],config["total"])
        
    #possible some callbacks
    callbacks = None
    return model,data_model,callbacks


# In[10]:


with open("configs/config_large_ds_climate.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


# In[11]:


print(config)


# In[12]:


blocks = config['blocks']
dt = config['dt']
dx = config['dx']
epochs = config['epochs']
lr = config['learning_rate']
opti = config['optimizer']


# In[13]:


model,data_model,callbacks = setenv(config)


# In[14]:


def getOptimizer(config):
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if config['optimizer'] == 'LBFGS':
        optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=10, line_search_fn=True,batch_mode=True)

    return optimizer


# In[15]:


##optimizer
optimizer = getOptimizer(config)
decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


# In[12]:


print(data_model.n_batch)


# In[ ]:


for block in blocks:
    print('[PRINT] block:',block)
    if block==0:
        print('[PRINT] Warmum Stage')
    stepnum = block if block>=1 else 1
    #get the data at this time #shape [block,batch,channel,X_dim]
    
    for epoch in range(epochs):
        print("epochs")
        #for every batch
        for b in range(data_model.n_batch):
            u_obs,non_eqn_dep_t,non_eqn_t,diff_dict = data_model.data(stepnum+1) #np array of stepnum elements 
            
            def closure():
                optimizer.zero_grad()
                loss,data_loss,syment_reg,moment_loss,loss_mse = modelLoss(model,u_obs,non_eqn_dep_t,non_eqn_t,diff_dict,config,block)
                
                if loss.requires_grad:
                        loss.backward()
                        
                return loss
            def closureTemp():
                optimizer.zero_grad()
                loss,data_loss,syment_reg,moment_loss,loss_mse = modelLoss(model,u_obs,non_eqn_dep_t,non_eqn_t,diff_dict,config,block)
                loss.backward()
                return loss,data_loss,syment_reg,moment_loss,loss_mse
            print("epochs 2")
            
            optimizer.step(closure)
            print("epochs 3")
            
            if b%10==0 and data_model.n_batch!=1:
                loss,data_loss,syment_reg,moment_loss,loss_mse = closureTemp()         
                print("[PRINT] Epoch: %d, Batch: %d, Loss: %.3f, Mse Loss: %.3f, Data Loss: %.3f, Symnet Regularize: %.3f, Moment Regularize: %.3f "                      % (epoch,b,loss,loss_mse,                          data_loss,syment_reg,                          moment_loss))

        if epoch%10==0:
            loss,data_loss,syment_reg,moment_loss,loss_mse = closureTemp()
            print("[PRINT] Epoch: %d, Loss: %.3f, Mse Loss: %.3f, Data Loss: %.3f, Symnet Regularize: %.3f, Moment Regularize: %.3f "                  % (epoch,loss,loss_mse,                      data_loss,syment_reg,                      moment_loss))
            
        if epoch%500==0:
            name = "Block_"+str(block)+"_Epoch_"+str(epoch)+".pth"
            torch.save(model.state_dict(),config['model_dir']+name)
            
    my_lr_scheduler.step() #this doesn't matter for LBFGS
    name = "Block_"+str(block)+"_Epoch_"+str(epochs)+".pth"
    torch.save(model.state_dict(),config['model_dir']+name)


# In[16]:


# u_obs,non_eqn_dep_t,non_eqn_t,diff_dict = data_model.data(1+1) #np array of stepnum elements 


# In[ ]:


# loss,data_loss,syment_reg,moment_loss,loss_mse = modelLoss(model,u_obs,non_eqn_dep_t,non_eqn_t,diff_dict,config,0)


# In[ ]:




