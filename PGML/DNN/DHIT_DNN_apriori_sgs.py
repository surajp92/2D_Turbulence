# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
from utils import *
import os
import time as tm
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs='?', default="dnn.txt", help="Config yaml file")
parser.add_argument("tf_version", nargs='?', default=1, type=int, help="Tensorflow version")
parser.add_argument("log", nargs='?', default=0, type=int, help="Write to a log file")
args = parser.parse_args()
config_file = args.config
tf_version = args.tf_version
print_log = args.log

if tf_version == 1:
    from keras.models import Sequential, Model, load_model
    from keras.layers import Dense, Dropout, Input
    from keras.callbacks import ModelCheckpoint
    #from keras.utils import plot_model
    from keras import optimizers
    from keras import backend as K

elif tf_version == 2:
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.callbacks import ModelCheckpoint
    #from keras.utils import plot_model
    from tensorflow.keras import optimizers
    from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

#%%
#Class of problem to solve 2D decaying homogeneous isotrpic turbulence
class DHIT:
    def __init__(self,nx,ny,nxf,nyf,re,freq,n_snapshots,n_snapshots_train,n_snapshots_test,
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
        self.re = re
        self.freq = freq
        self.n_snapshots = n_snapshots
        self.n_snapshots_train = n_snapshots_train
        self.n_snapshots_test = n_snapshots_test
        self.istencil = istencil
        self.ifeatures = ifeatures
        self.ilabel = ilabel
        
        self.max_min =  np.zeros(shape=(15, 2), dtype='double')
        
        self.wc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.sc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        
        self.pi = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        
        if self.ifeatures >= 2:
            self.kwc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
            self.ksc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
            if self.ifeatures == 3:                
                self.wcx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.wcy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.wcxx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.wcyy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.wcxy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.scx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.scy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.scxx = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.scyy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
                self.scxy = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        
        directory = f'../KT_DNS/solution_{nxf}_{nx}_{re:0.2e}/apriori'
        
        for m in range(1,self.n_snapshots+1):    
            
            file_input = os.path.join(directory, f'ws_{m}.npz') 
            data_input = np.load(file_input)
            self.wc[m-1,:,:] = data_input['wc']
            self.sc[m-1,:,:] = data_input['sc']
            self.pi[m-1,:,:] = data_input['pi']
            if self.ifeatures >= 2:
                self.kwc[m-1,:,:] = data_input['kw']
                self.ksc[m-1,:,:] = data_input['ks']
                if self.ifeatures == 3:         
                    self.wcx[m-1,:,:] = data_input['wcx']
                    self.wcy[m-1,:,:] = data_input['wcy']
                    self.wcxx[m-1,:,:] = data_input['wcxx']
                    self.wcyy[m-1,:,:] = data_input['wcyy']
                    self.wcxy[m-1,:,:] = data_input['wcxy']
                    self.scx[m-1,:,:] = data_input['scx']
                    self.scy[m-1,:,:] = data_input['scy']
                    self.scxx[m-1,:,:] = data_input['scxx']
                    self.scyy[m-1,:,:] = data_input['scyy']
                    self.scxy[m-1,:,:] = data_input['scxy']
        
        self.scale_data()
        
        self.x_train,self.y_train = self.gen_train_data()
        
        self.x_test,self.y_test = self.gen_test_data()
    
    def scale_data(self):
        '''
        scaling the data between (-1,1) using (2x-(xmax+xmin))/(xmax-xmin)

        '''
        self.max_min[0,0], self.max_min[0,1] = np.max(self.wc), np.min(self.wc)
        self.max_min[1,0], self.max_min[1,1] = np.max(self.sc), np.min(self.sc)
        
        if self.ifeatures >= 2:
            self.max_min[2,0], self.max_min[2,1] = np.max(self.kwc), np.min(self.kwc)
            self.max_min[3,0], self.max_min[3,1] = np.max(self.ksc), np.min(self.ksc)
            
            if self.ifeatures == 3:
                self.max_min[4,0], self.max_min[4,1] = np.max(self.wcx), np.min(self.wcx)
                self.max_min[5,0], self.max_min[5,1] = np.max(self.wcy), np.min(self.wcy)
                self.max_min[6,0], self.max_min[6,1] = np.max(self.wcxx), np.min(self.wcxx)
                self.max_min[7,0], self.max_min[7,1] = np.max(self.wcyy), np.min(self.wcyy)
                self.max_min[8,0], self.max_min[8,1] = np.max(self.wcxy), np.min(self.wcxy)
                
                self.max_min[9,0], self.max_min[9,1] = np.max(self.scx), np.min(self.scx)
                self.max_min[10,0], self.max_min[10,1] = np.max(self.scy), np.min(self.scy)
                self.max_min[11,0], self.max_min[11,1] = np.max(self.scxx), np.min(self.scxx)
                self.max_min[12,0], self.max_min[12,1] = np.max(self.scyy), np.min(self.scyy)
                self.max_min[13,0], self.max_min[13,1] = np.max(self.scxy), np.min(self.scxy)
        
        self.max_min[14,0], self.max_min[14,1] = np.max(self.pi), np.min(self.pi)
        
        self.wc = (2.0*self.wc - (np.max(self.wc) + np.min(self.wc)))/(np.max(self.wc) - np.min(self.wc))
        self.sc = (2.0*self.sc - (np.max(self.sc) + np.min(self.sc)))/(np.max(self.sc) - np.min(self.sc))
        
        if self.ifeatures >= 2:
            self.kwc = (2.0*self.kwc - (np.max(self.kwc) + np.min(self.kwc)))/(np.max(self.kwc) - np.min(self.kwc))
            self.ksc = (2.0*self.ksc - (np.max(self.ksc) + np.min(self.ksc)))/(np.max(self.ksc) - np.min(self.ksc))
            
            if self.ifeatures == 3:
                self.wcx = (2.0*self.wcx - (np.max(self.wcx) + np.min(self.wcx)))/(np.max(self.wcx) - np.min(self.wcx))
                self.wcy = (2.0*self.wcy - (np.max(self.wcy) + np.min(self.wcy)))/(np.max(self.wcy) - np.min(self.wcy))
                self.wcxx = (2.0*self.wcxx - (np.max(self.wcxx) + np.min(self.wcxx)))/(np.max(self.wcxx) - np.min(self.wcxx))
                self.wcyy = (2.0*self.wcyy - (np.max(self.wcyy) + np.min(self.wcyy)))/(np.max(self.wcyy) - np.min(self.wcyy))
                self.wcxy = (2.0*self.wcxy - (np.max(self.wcxy) + np.min(self.wcxy)))/(np.max(self.wcxy) - np.min(self.wcxy))
                
                self.scx = (2.0*self.scx - (np.max(self.scx) + np.min(self.scx)))/(np.max(self.scx) - np.min(self.scx))
                self.scy = (2.0*self.scy - (np.max(self.scy) + np.min(self.scy)))/(np.max(self.scy) - np.min(self.scy))
                self.scxx = (2.0*self.scxx - (np.max(self.scxx) + np.min(self.scxx)))/(np.max(self.scxx) - np.min(self.scxx))
                self.scyy = (2.0*self.scyy - (np.max(self.scyy) + np.min(self.scyy)))/(np.max(self.scyy) - np.min(self.scyy))
                self.scxy = (2.0*self.scxy - (np.max(self.scxy) + np.min(self.scxy)))/(np.max(self.scxy) - np.min(self.scxy))
        
        self.pi = (2.0*self.pi - (np.max(self.pi) + np.min(self.pi)))/(np.max(self.pi) - np.min(self.pi))
        
    
        
    def gen_train_data(self):
        
        '''
        data generation for training and testing CNN model

        '''
        
        # train data
        for p in range(1,self.n_snapshots_train+1):            
            m = p*self.freq
            nx,ny = self.nx, self.ny
            nt = int((nx+1)*(ny+1))
            
            if self.istencil == 1 and self.ifeatures == 1:
                x_t = np.zeros((nt,2))
            elif self.istencil == 1 and self.ifeatures == 2:
                x_t = np.zeros((nt,4))
            elif self.istencil == 1 and self.ifeatures == 3:
                x_t = np.zeros((nt,12))
            elif self.istencil == 2 and self.ifeatures == 1:
                x_t = np.zeros((nt,18))
            elif self.istencil == 2 and self.ifeatures == 2:
                x_t = np.zeros((nt,36))
            elif self.istencil == 2 and self.ifeatures == 3:
                x_t = np.zeros((nt,108))
                            
            if self.ilabel == 1:
                y_t = np.zeros((nt,1))
            elif self.ilabel == 2:
                y_t = np.zeros((nt,3))
            
            n = 0
            
            if self.istencil == 1 and self.ifeatures == 1:
                x_t[:,0] = self.wc[m-1,:,:].flatten()
                x_t[:,1] = self.sc[m-1,:,:].flatten()
                
            elif self.istencil == 1 and self.ifeatures == 2:
                x_t[:,0] = self.wc[m-1,:,:].flatten()
                x_t[:,1] = self.sc[m-1,:,:].flatten()
                x_t[:,2] = self.kwc[m-1,:,:].flatten()
                x_t[:,3] = self.ksc[m-1,:,:].flatten()

            elif self.istencil == 1 and self.ifeatures == 3:
                x_t[:,0] = self.wc[m-1,:,:].flatten()
                x_t[:,1] = self.sc[m-1,:,:].flatten()
                x_t[:,2] = self.wcx[m-1,:,:].flatten()
                x_t[:,3] = self.wcy[m-1,:,:].flatten()
                x_t[:,4] = self.wcxx[m-1,:,:].flatten()
                x_t[:,5] = self.wcyy[m-1,:,:].flatten()
                x_t[:,6] = self.wcxy[m-1,:,:].flatten()
                x_t[:,7] = self.scx[m-1,:,:].flatten()
                x_t[:,8] = self.scy[m-1,:,:].flatten()
                x_t[:,9] = self.scxx[m-1,:,:].flatten()
                x_t[:,10] = self.scyy[m-1,:,:].flatten()
                x_t[:,11] = self.scxy[m-1,:,:].flatten()

            elif self.istencil == 2 and self.ifeatures == 1:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                        n = n+1

            elif self.istencil == 2 and self.ifeatures == 2:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,18:27] = self.kwc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,27:36] = self.ksc[m-1,i-1:i+2,j-1:j+2].flatten()
                        n = n+1
              
            elif self.istencil == 2 and self.ifeatures == 3:
                for i in range(1,nx):
                    for j in range(1,ny):
                        x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,18:27] = self.wcx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,27:36] = self.wcy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,36:45] = self.wcxx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,45:54] = self.wcyy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,54:63] = self.wcxy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,63:72] = self.scx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,72:81] = self.scy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,81:90] = self.scxx[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,90:99] = self.scyy[m-1,i-1:i+2,j-1:j+2].flatten()
                        x_t[n,99:108] = self.scxy[m-1,i-1:i+2,j-1:j+2].flatten()
                        n = n+1
            
            n = 0
            if self.ilabel == 1:
                y_t[:,0] = self.pi[m-1,:,:].flatten()

            elif self.ilabel == 2:
                y_t[:,0] = self.t11[m-1,:,:].flatten()
                y_t[:,1] = self.t12[m-1,:,:].flatten()
                y_t[:,2] = self.t22[m-1,:,:].flatten()
                                            
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
        nt = int((nx+1)*(ny+1))
        
        if self.istencil == 1 and self.ifeatures == 1:
            x_t = np.zeros((nt,2))
        elif self.istencil == 1 and self.ifeatures == 2:
            x_t = np.zeros((nt,4))
        elif self.istencil == 1 and self.ifeatures == 3:
            x_t = np.zeros((nt,12))
        elif self.istencil == 2 and self.ifeatures == 1:
            x_t = np.zeros((nt,18))
        elif self.istencil == 2 and self.ifeatures == 2:
            x_t = np.zeros((nt,36))
        elif self.istencil == 2 and self.ifeatures == 3:
            x_t = np.zeros((nt,108))
                
        if self.ilabel == 1:
            y_t = np.zeros((nt,1))
        elif self.ilabel == 2:
            y_t = np.zeros((nt,3))
        
        n = 0
            
        if self.istencil == 1 and self.ifeatures == 1:
            x_t[:,0] = self.wc[m-1,:,:].flatten()
            x_t[:,1] = self.sc[m-1,:,:].flatten()
            
        elif self.istencil == 1 and self.ifeatures == 2:
                x_t[:,0] = self.wc[m-1,:,:].flatten()
                x_t[:,1] = self.sc[m-1,:,:].flatten()
                x_t[:,2] = self.kwc[m-1,:,:].flatten()
                x_t[:,3] = self.ksc[m-1,:,:].flatten()

        elif self.istencil == 1 and self.ifeatures == 3:
            x_t[:,0] = self.wc[m-1,:,:].flatten()
            x_t[:,1] = self.sc[m-1,:,:].flatten()
            x_t[:,2] = self.wcx[m-1,:,:].flatten()
            x_t[:,3] = self.wcy[m-1,:,:].flatten()
            x_t[:,4] = self.wcxx[m-1,:,:].flatten()
            x_t[:,5] = self.wcyy[m-1,:,:].flatten()
            x_t[:,6] = self.wcxy[m-1,:,:].flatten()
            x_t[:,7] = self.scx[m-1,:,:].flatten()
            x_t[:,8] = self.scy[m-1,:,:].flatten()
            x_t[:,9] = self.scxx[m-1,:,:].flatten()
            x_t[:,10] = self.scyy[m-1,:,:].flatten()
            x_t[:,11] = self.scxy[m-1,:,:].flatten()
        
        elif self.istencil == 2 and self.ifeatures == 1:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                    n = n+1

        elif self.istencil == 2 and self.ifeatures == 2:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,18:27] = self.kwc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,27:36] = self.ksc[m-1,i-1:i+2,j-1:j+2].flatten()
                    n = n+1
          
        elif self.istencil == 2 and self.ifeatures == 3:
            for i in range(1,nx):
                for j in range(1,ny):
                    x_t[n,0:9] = self.wc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,9:18] = self.sc[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,18:27] = self.wcx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,27:36] = self.wcy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,36:45] = self.wcxx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,45:54] = self.wcyy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,54:63] = self.wcxy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,63:72] = self.scx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,72:81] = self.scy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,81:90] = self.scxx[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,90:99] = self.scyy[m-1,i-1:i+2,j-1:j+2].flatten()
                    x_t[n,99:108] = self.scxy[m-1,i-1:i+2,j-1:j+2].flatten()
                    n = n+1
        
        n = 0
        if self.ilabel == 1:
                y_t[:,0] = self.pi[m-1,:,:].flatten()

        elif self.ilabel == 2:
            y_t[:,0] = self.t11[m-1,:,:].flatten()
            y_t[:,1] = self.t12[m-1,:,:].flatten()
            y_t[:,2] = self.t22[m-1,:,:].flatten()
                    
        x_test = x_t
        y_test = y_t
        
        return x_test, y_test
    
    
#%%
#A Convolutional Neural Network class
class DNN:
    def __init__(self,x_train,y_train,x_valid,y_valid,nf,nl,n_layers,n_neurons,lr):
        
        '''
        initialize the CNN class
        
        Inputs
        ------
        x_train : input features of the DNN model
        y_train : output label of the DNN model
        nf : number of input features
        nl : number of output labels
        '''
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.nf = nf
        self.nl = nl
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.lr = lr
        self.model = self.DNN(x_train,y_train,nf,nl)
    
    def coeff_determination(self,y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
    def DNN(self,x_train,y_train,nf,nl):
        
        '''
        define CNN model
        
        Inputs
        ------
        x_train : input features of the DNN model
        y_train : output label of the DNN model
        nf : number of input features
        nl : number of output labels
        
        Output
        ------
        model: DNN model with defined activation function, number of layers
        '''
        
        # L2 regularization: kernel_regularizer=l2(0.001)
        model = Sequential()
        #model.add(Dropout(0.2))
        input_layer = Input(shape=(self.nf,))
        
        x = Dense(self.n_neurons[0], activation='relu',  use_bias=True)(input_layer)
        for i in range(1,self.n_layers):
            x = Dense(self.n_neurons[i], activation='relu',  use_bias=True)(x)
        
        output_layer = Dense(nl, activation='linear', use_bias=True)(x)
        
        model = Model(input_layer, output_layer)
        
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=adam)#, metrics=[self.coeff_determination])
        
        return model

    def DNN_compile(self):
        
        '''
        compile the CNN model
        
        Inputs
        ------
        optimizer: optimizer of the DNN

        '''
        
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=[self.coeff_determination])
        
    def DNN_train(self,epochs,batch_size,filename):
        
        '''
        train the CNN model
        
        Inputs
        ------
        epochs: number of epochs of the training
        batch_size: batch size of the training
        
        Output
        ------
        history_callback: return the loss history of CNN model training
        '''
        
        filepath = filename
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        history_callback = self.model.fit(self.x_train,self.y_train,
                                          epochs=epochs,batch_size=batch_size, 
                                          validation_data= (self.x_valid,self.y_valid),
                                          callbacks=callbacks_list)
        return history_callback
    
    def DNN_history(self, history_callback):
        
        '''
        get the training and validation loss history
        
        Inputs
        ------
        history_callback: loss history of DNN model training
        
        Output
        ------
        loss: training loss history of DNN model training
        val_loss: validation loss history of DNN model training
        '''
        
        loss = history_callback.history["loss"]
        val_loss = history_callback.history["val_loss"]
        mse = history_callback.history['coeff_determination']
        val_mse = history_callback.history['val_coeff_determination']
        
        return loss, val_loss, mse, val_mse
            
    def DNN_predict(self,x_test):
        
        '''
        predict the label for input features
        
        Inputs
        ------
        x_test: test data (has same shape as input features used for training)
        
        Output
        ------
        y_test: predicted output by the CNN (has same shape as label used for training)
        '''
        
        testing_time_init1 = tm.time()
        y_test = self.model.predict(x_test)
        t1 = tm.time() - testing_time_init1
        
        testing_time_init2 = tm.time()
        y_test = self.model.predict(x_test)
        #y_test = custom_model.predict(x_test)
        t2 = tm.time() - testing_time_init2
        
        testing_time_init3 = tm.time()
        y_test = self.model.predict(x_test)
        y_test = self.model.predict(x_test)
        t3 = tm.time() - testing_time_init3
        
        return y_test,t1,t2,t3
    
    def DNN_predict1(self,x_test,ist,ift,nsm):
        
        '''
        predict the label for input features
        
        Inputs
        ------
        x_test: test data (has same shape as input features used for training)
        
        Output
        ------
        y_test: predicted output by the CNN (has same shape as label used for training)
        '''
        filepath = 'tcfd_paper_data/new_data_sgs/ann_'+str(ist)+'_'+str(ift)+'_'+str(nsm)
        
        custom_model = load_model(filepath+'/dnn_best_model.hd5', 
                                  custom_objects={'coeff_determination': self.coeff_determination})
                                  
        
        testing_time_init1 = tm.time()
        y_test = custom_model.predict(x_test)
        t1 = tm.time() - testing_time_init1
        
        testing_time_init2 = tm.time()
        y_test = custom_model.predict(x_test)
        #y_test = custom_model.predict(x_test)
        t2 = tm.time() - testing_time_init2
        
        testing_time_init3 = tm.time()
        y_test = custom_model.predict(x_test)
        y_test = custom_model.predict(x_test)
        t3 = tm.time() - testing_time_init3
        
        return y_test,t1,t2,t3
    
    def DNN_info(self):
        
        '''
        print the CNN model summary
        '''
        
        self.model.summary()
        #plot_model(self.model, to_file='dnn_model.png')
     
        
#%%
# generate training and testing data for CNN
l1 = []
with open(config_file) as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nxf, nyf = np.int64(l1[0][0]), np.int64(l1[0][0])
nx, ny = np.int64(l1[1][0]), np.int64(l1[1][0])
n_snapshots = np.int64(l1[2][0])
n_snapshots_train = np.int64(l1[3][0])   
n_snapshots_test = np.int64(l1[4][0])        
freq = np.int64(l1[5][0])
istencil = np.int64(l1[6][0])    
ifeatures = np.int64(l1[7][0])   
ilabel = np.int64(l1[8][0])      
re = np.float64(l1[9][0]) 

if tf_version == 1:
    directory = f'nn_history/TF1_{nx}/'
elif tf_version == 2:
    directory = f'nn_history/TF2_{nx}/'
    
if not os.path.exists(directory):
    os.makedirs(directory)
    
# hyperparameters initilization
n_layers = 2
n_neurons = [40,40]
lr = 0.001

#%%
obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,re=re,freq=freq,
           n_snapshots=n_snapshots,n_snapshots_train=n_snapshots_train, 
           n_snapshots_test=n_snapshots_test,
           istencil=istencil,ifeatures=ifeatures,ilabel=ilabel)

max_min = obj.max_min

data_sc,labels_sc = obj.x_train,obj.y_train
x_test_sc,y_test_sc = obj.x_test,obj.y_test


#%%
x_train, x_valid, y_train, y_valid = train_test_split(data_sc, labels_sc, test_size=0.2 , shuffle= True)

ns_train,nf = x_train.shape
ns_train,nl = y_train.shape 

#%%
# train the CNN model and predict for the test data
model = DNN(x_train,y_train,x_valid,y_valid,nf,nl,n_layers,n_neurons,lr)
model.DNN_info()
model.DNN_compile()

#%%
training_time_init = tm.time()

if tf_version == 1:
    filename = os.path.join(directory, f'DNN_model_{istencil}_{ifeatures}')    
elif tf_version == 2:
    filename = os.path.join(directory, f'DNN_model_{istencil}_{ifeatures}')   
    
history_callback = model.DNN_train(epochs=100,batch_size=1024,filename=filename)

total_training_time = tm.time() - training_time_init

filename = os.path.join(directory, f'scaling.npy')
np.save(filename,max_min)

loss, val_loss, mse, val_mse = model.DNN_history(history_callback)
nn_history(loss, val_loss, mse, val_mse, istencil, ifeatures, n_snapshots_train, directory)

#%%
total_training_time =0
y_pred_sc, t1, t2, t3 = model.DNN_predict(x_test_sc)

with open('cpu_time.csv', 'a', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(['DNN',istencil, ifeatures, n_snapshots_train, total_training_time, t1, t2, t3])
     
#export_resutls(y_test, y_pred, ilabel, istencil, ifeatures, n_snapshots_train, nxf, nx, nn = 1)

#%%
if ilabel == 1:
    y_test = np.zeros(shape=((nx+1)*(ny+1), 1), dtype='double') 
    y_pred = np.zeros(shape=((nx+1)*(ny+1), 1), dtype='double')
    for i in range(1):
        y_pred[:,i] = 0.5*(y_pred_sc[:,i]*(max_min[-1,0] - max_min[-1,1]) + (max_min[-1,0] + max_min[-1,1]))
        y_test[:,i] = 0.5*(y_test_sc[:,i]*(max_min[-1,0] - max_min[-1,1]) + (max_min[-1,0] + max_min[-1,1]))    

elif ilabel == 2:
    y_test = np.zeros(shape=((nx+1)*(ny+1), 3), dtype='double') 
    y_pred = np.zeros(shape=((nx+1)*(ny+1), 3), dtype='double')
    for i in range(3):
        y_pred[:,i] = 0.5*(y_pred_sc[:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))
        y_test[:,i] = 0.5*(y_test_sc[:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))   
      
if ilabel == 1:
    filename = os.path.join(directory, f'y_sgs_{istencil}_{ifeatures}_{n_snapshots_train}_{ilabel}.npz')
    np.savez(filename, y_test=y_test, y_pred=y_pred)
if ilabel == 2:
    filename = os.path.join(directory, f'y_sgs_{istencil}_{ifeatures}_{n_snapshots_train}_{ilabel}.npz')
    np.savez(filename, y_test=y_test, y_pred=y_pred)


#%%
# histogram plot for shear stresses along with probability density function 
# PDF formula: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
if ilabel == 1:
    num_bins = 64
    
    fig, axs = plt.subplots(1,1,figsize=(6,4))
    axs.set_yscale('log')

    
    # the histogram of the data
    axs.hist(y_test[:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),
               density=True,label="True")
        
    axs.hist(y_pred[:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),
               density=True,label="CNN")  
    
    x_ticks = np.arange(-4.1*np.std(y_test[:,0]), 4.1*np.std(y_test[:,0]), np.std(y_test[:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs.set_xlabel(r"$\Pi$")
    axs.set_ylabel("PDF")
    axs.set_xticks(x_ticks)                                                           
    axs.set_xticklabels(x_labels)              
           
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["True",  "DNN"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()
    
    filename = os.path.join(directory, f'ts_dnn_{istencil}_{ifeatures}_{n_snapshots_train}.png')    
    fig.savefig(filename, bbox_inches = 'tight', dpi=200)
    
elif ilabel == 2:
    num_bins = 64
    
    fig, axs = plt.subplots(1,3,figsize=(13,4))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    
    # the histogram of the data
    axs[0].hist(y_test[:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),
                                     density=True,label="True")
    
    axs[0].hist(t11s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label=r"Dynamic")
    
    axs[0].hist(y_pred[:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),
                                     density=True,label="DNN")
    
    x_ticks = np.arange(-4*np.std(y_test[:,0]), 4.1*np.std(y_test[:,0]), np.std(y_test[:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[0].set_xlabel(r"$\tau_{11}$")
    axs[0].set_ylabel("PDF")
    axs[0].set_xticks(x_ticks)                                                           
    axs[0].set_xticklabels(x_labels)              
    
    #------#
    axs[1].hist(y_test[:,1].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),
                                     density=True,label="True")
    
    axs[1].hist(t12s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label=r"Dynamic")
    
    axs[1].hist(y_pred[:,1].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),
                                     density=True,label="DNN")
    
    x_ticks = np.arange(-4*np.std(y_test[:,1]), 4.1*np.std(y_test[:,1]), np.std(y_test[:,1]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[1].set_xlabel(r"$\tau_{12}$")
    #axs[1].set_ylabel("PDF")
    axs[1].set_xticks(x_ticks)                                                           
    axs[1].set_xticklabels(x_labels)              
    
    #------#
    axs[2].hist(y_test[:,2].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),
                                     density=True,label="True")
    
    axs[2].hist(t22s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label=r"Dynamic")
    
    axs[2].hist(y_pred[:,2].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),
                                     density=True,label="DNN")
    
    x_ticks = np.arange(-4*np.std(y_test[:,2]), 4.1*np.std(y_test[:,2]), np.std(y_test[:,2]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[2].set_xlabel(r"$\tau_{22}$")
    #axs[2].set_ylabel("PDF")
    axs[2].set_xticks(x_ticks)                                                           
    axs[2].set_xticklabels(x_labels)              
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["True", "DSM", "ANN"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()
    
    fig.savefig('nn_history/ts_dnn_'+str(istencil)+'_'+str(ifeatures)+'_'+str(n_snapshots_train)+'.png', bbox_inches = 'tight')
