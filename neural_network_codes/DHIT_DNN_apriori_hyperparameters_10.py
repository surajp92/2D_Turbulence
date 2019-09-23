# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import optimizers
from scipy.stats import norm 
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from keras.regularizers import l2

from utils import *
import os

#%%
#Class of problem to solve 2D decaying homogeneous isotrpic turbulence
class DHIT:
    def __init__(self,nx,ny,nxf,nyf,freq,n_snapshots,n_snapshots_train,n_snapshots_test,
                 istencil,ifeatures,ilabel):
        
        '''
        initialize the DHIT class
        
        Inputs
        ------
        n_snapshots : number of snapshots available
        nx,ny : dimension of the snapshot

        '''
        
        self.nx = nx
        self.ny = ny
        self.nxf = nxf
        self.nyf = nyf
        self.freq = freq
        self.n_snapshots = n_snapshots
        self.n_snapshots_train = n_snapshots_train
        self.n_snapshots_test = n_snapshots_test
        self.istencil = istencil
        self.ifeatures = ifeatures
        self.ilabel = ilabel
        
        self.uc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.vc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.ucx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.ucy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.vcx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.vcy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.ucxx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.ucyy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.vcxx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.vcyy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.t11 = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.t12 = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.t22 = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.nu = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        
        for m in range(1,self.n_snapshots+1):
            folder = "data_"+ str(self.nxf) + "_" + str(self.nx) 
            
            file_input = "../data_spectral/"+folder+"/uc/uc_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.uc[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/vc/vc_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.vc[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/ucx/ucx_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.ucx[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/ucy/ucy_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.ucy[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/vcx/vcx_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.vcx[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/vcy/vcy_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.vcy[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/ucxx/ucxx_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.ucxx[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/ucyy/ucyy_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.ucyy[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/vcxx/vcxx_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.vcxx[m-1,:,:] = data_input
            
            file_input = "../data_spectral/"+folder+"/vcyy/vcyy_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.vcyy[m-1,:,:] = data_input
            
            file_output = "../data_spectral/"+folder+"/true_shear_stress/t_"+str(m)+".csv"
            data_output = np.genfromtxt(file_output, delimiter=',')
            data_output = data_output.reshape((3,self.nx+1,self.ny+1))
            self.t11[m-1,:,:] = data_output[0,:,:]
            self.t12[m-1,:,:] = data_output[1,:,:]
            self.t22[m-1,:,:] = data_output[2,:,:]
            
            file_input = "../data_spectral/"+folder+"/nu_smag/nus_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            self.nu[m-1,:,:] = data_input
            
        self.x_train,self.y_train = self.gen_train_data()
        self.x_test,self.y_test = self.gen_test_data()
        
    def gen_train_data(self):
        
        '''
        data generation for training and testing CNN model

        '''
        
        # train data
        for p in range(1,self.n_snapshots_train+1): 
            m = p*self.freq 
            nx,ny = self.nx, self.ny
            nt = int((nx-1)*(ny-1))
            
            if self.istencil == 1 and self.ifeatures == 1:
                x_t = np.zeros((nt,90))
            elif self.istencil == 1 and self.ifeatures == 2:
                x_t = np.zeros((nt,18))
            elif self.istencil == 2 and self.ifeatures == 1:
                x_t = np.zeros((nt,10))
            elif self.istencil == 2 and self.ifeatures == 2:
                x_t = np.zeros((nt,2))
            
            if self.ilabel == 1:
                y_t = np.zeros((nt,3))
            elif self.ilabel == 2:
                y_t = np.zeros((nt,1))
            
            n = 0
            
            if istencil == 1 and ifeatures == 1:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.uc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,9:18] = self.vc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,18:27] = self.ucx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,27:36] = self.ucy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,36:45] = self.vcx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,45:54] = self.vcy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,54:63] = self.ucxx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,63:72] = self.ucyy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,72:81] = self.vcxx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,81:90] = self.vcyy[m-1,i-1:i+2,j-1:j+2].flatten()
                        n = n+1
            
            elif istencil == 1 and ifeatures == 2:  
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.uc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,9:18] = self.vc[m-1,i-1:i+2,j-1:j+2].flatten()
                        n = n+1
            
            elif istencil == 2 and ifeatures == 1:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0] = self.uc[m-1,i,j]
                        x_t[n,1] = self.vc[m-1,i,j]
                        x_t[n,2] = self.ucx[m-1,i,j]
                        x_t[n,3] = self.ucy[m-1,i,j]
                        x_t[n,4] = self.vcx[m-1,i,j]
                        x_t[n,5] = self.vcy[m-1,i,j]
                        x_t[n,6] = self.ucxx[m-1,i,j]
                        x_t[n,7] = self.ucyy[m-1,i,j]
                        x_t[n,8] = self.vcxx[m-1,i,j]
                        x_t[n,9] = self.vcyy[m-1,i,j]
                        n = n+1
                                                
            elif istencil == 2 and ifeatures == 2:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.uc[m-1,i,j]
                        x_t[n,9:18] = self.vc[m-1,i,j]
                        n = n+1
            
            n = 0
            if ilabel == 1:
                for i in range(1,nx):
                    for j in range(1,ny):
                        y_t[n,0], y_t[n,1], y_t[n,2] = self.t11[m-1,i,j], self.t12[m-1,i,j], self.t22[m-1,i,j]
                        n = n+1
                        
            elif ilabel == 2:
                for i in range(1,nx):
                    for j in range(1,ny):
                        y_t[n,0] = self.nu[m-1,i,j]
                        n = n+1       
            
            if p == 1:
                x_train = x_t
                y_train = y_t
            else:
                x_train = np.vstack((x_train,x_t))
                y_train = np.vstack((y_train,y_t))
        
        return x_train, y_train
    
    def gen_test_data(self):
        
        # test data
        m = self.n_snapshots_test
        nx,ny = self.nx, self.ny
        nt = int((nx-1)*(ny-1))

        if self.istencil == 1 and self.ifeatures == 1:
                x_t = np.zeros((nt,90))
        elif self.istencil == 1 and self.ifeatures == 2:
            x_t = np.zeros((nt,18))
        elif self.istencil == 2 and self.ifeatures == 1:
            x_t = np.zeros((nt,10))
        elif self.istencil == 2 and self.ifeatures == 2:
            x_t = np.zeros((nt,2))
        
        if self.ilabel == 1:
            y_t = np.zeros((nt,3))
        elif self.ilabel == 2:
            y_t = np.zeros((nt,1))
        
        n = 0
        
        if istencil == 1 and ifeatures == 1:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.uc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,9:18] = self.vc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,18:27] = self.ucx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,27:36] = self.ucy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,36:45] = self.vcx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,45:54] = self.vcy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,54:63] = self.ucxx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,63:72] = self.ucyy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,72:81] = self.vcxx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,81:90] = self.vcyy[m-1,i-1:i+2,j-1:j+2].flatten()
                    n = n+1
            
        elif istencil == 1 and ifeatures == 2:  
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.uc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,9:18] = self.vc[m-1,i-1:i+2,j-1:j+2].flatten()
                    n = n+1
        
        elif istencil == 2 and ifeatures == 1:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0] = self.uc[m-1,i,j]
                    x_t[n,1] = self.vc[m-1,i,j]
                    x_t[n,2] = self.ucx[m-1,i,j]
                    x_t[n,3] = self.ucy[m-1,i,j]
                    x_t[n,4] = self.vcx[m-1,i,j]
                    x_t[n,5] = self.vcy[m-1,i,j]
                    n = n+1
                                            
        elif istencil == 2 and ifeatures == 2:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.uc[m-1,i,j]
                    x_t[n,9:18] = self.vc[m-1,i,j]
                    n = n+1
        
        n = 0
        if ilabel == 1:
            for i in range(1,nx):
                for j in range(1,ny):
                    y_t[n,0], y_t[n,1], y_t[n,2] = self.t11[m-1,i,j], self.t12[m-1,i,j], self.t22[m-1,i,j]
                    n = n+1
                    
        elif ilabel == 2:
            for i in range(1,nx):
                for j in range(1,ny):
                    y_t[n,0] = self.nu[m-1,i,j]
                    n = n+1   
        
        x_test = x_t
        y_test = y_t
        
        return x_test, y_test
       
#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_dnn_model(n_layers=2, n_neurons=60, lr=0.001, n_features=90, n_labels=3):
    model = Sequential()
    
    input_layer = Input(shape=(n_features,))
    
    x = Dense(n_neurons, activation='relu',  use_bias=True)(input_layer)
    for i in range(1,n_layers):
        x = Dense(n_neurons, activation='relu',  use_bias=True)(x)
    
    output_layer = Dense(n_labels, activation='linear', use_bias=True)(x)
    
    model = Model(input_layer, output_layer)
    
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[coeff_determination])
    
    model.summary()
    
    return model

dnn_model = build_dnn_model()
dnn_model.summary()

#%%
model_dnn = KerasRegressor(
    build_fn = build_dnn_model,
    epochs=400, batch_size=256, verbose=-1)

def coeff_determination_score(y_true, y_pred):
    SS_res =  np.sum(np.square( y_true-y_pred ))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 1e-7) )

my_custom_scorer = make_scorer(coeff_determination_score, greater_is_better=True)

n_layers = [2,3,5,7]
n_neurons = [40,60,80,100]
lr_params = [0.001, 0.0001]

keras_param_options = {
    'n_layers': n_layers,
    'n_neurons': n_neurons,  
    'lr': lr_params
}

rs_lstm = GridSearchCV( 
    estimator = model_dnn, 
    param_grid = keras_param_options,
    scoring = my_custom_scorer,
    cv = 5,
    n_jobs = -25,
    #verbose = -1
)

#%%
# generate training and testing data for CNN
l1 = []
with open('dnn_10.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nxf, nyf = np.int64(l1[0][0]), np.int64(l1[0][0])
nx, ny = np.int64(l1[1][0]), np.int64(l1[1][0])
n_snapshots = np.int64(l1[2][0])
n_snapshots_train = np.int64(l1[3][0])   
n_snapshots_test = np.int64(l1[4][0])        
freq = np.int64(l1[5][0])
istencil = np.int64(l1[6][0])    # 1: nine point, 2: single point
ifeatures = np.int64(l1[7][0])   # 1: 6 features, 2: 2 features 
ilabel = np.int64(l1[8][0])      # 1: SGS (tau), 2: eddy-viscosity (nu)

#%%
obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,freq=freq,n_snapshots=n_snapshots,n_snapshots_train=n_snapshots_train, 
           n_snapshots_test=n_snapshots_test,istencil=istencil,ifeatures=ifeatures,ilabel=ilabel)

data,labels= obj.x_train,obj.y_train
x_test,y_test = obj.x_test,obj.y_test

#%%
# scaling between (-1,1)
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(data)
data_sc = sc_input.transform(data)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(labels)
labels_sc = sc_output.transform(labels)

#%%
rs_result = rs_lstm.fit(data_sc, labels_sc)

means = rs_result.cv_results_['mean_test_score']
stds = rs_result.cv_results_['std_test_score']
params = rs_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print('Best score obtained: {0}'.format(rs_lstm.best_score_))
print('Parameters:')
for param, value in rs_lstm.best_params_.items():
    print('\t{}: {}'.format(param, value))

