# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate
from keras.utils import plot_model
from keras import backend as K
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm 
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
from utils import *
import os
import time as tm
import csv

import os

#import pydot

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
        self.kwc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.ksc = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
        self.pi = np.zeros(shape=(self.n_snapshots, self.nx+1, self.ny+1), dtype='double')
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
            self.kwc[m-1,:,:] = data_input['kw']
            self.ksc[m-1,:,:] = data_input['ks']
            self.pi[m-1,:,:] = data_input['pi']
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
        self.max_min[2,0], self.max_min[2,1] = np.max(self.kwc), np.min(self.kwc)
        self.max_min[3,0], self.max_min[3,1] = np.max(self.ksc), np.min(self.ksc)
        
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
        self.kwc = (2.0*self.kwc - (np.max(self.kwc) + np.min(self.kwc)))/(np.max(self.kwc) - np.min(self.kwc))
        self.ksc = (2.0*self.ksc - (np.max(self.ksc) + np.min(self.ksc)))/(np.max(self.ksc) - np.min(self.ksc))
        
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
        
        if self.ifeatures == 1:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 2), dtype='double')
        elif self.ifeatures == 2:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 4), dtype='double')
        elif self.ifeatures == 3:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 12), dtype='double')
            
        if self.ilabel == 1:
            y_train = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 1), dtype='double')
        elif self.ilabel == 2:
            y_train = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 3), dtype='double')
        
        for m in range(1,self.n_snapshots_train+1):
            #m = p*self.freq
            if self.ifeatures == 1:
                x_train[m-1,:,:,0] = self.wc[m-1]
                x_train[m-1,:,:,1] = self.sc[m-1]
            if self.ifeatures == 2:
                x_train[m-1,:,:,0] = self.wc[m-1]
                x_train[m-1,:,:,1] = self.sc[m-1]
                x_train[m-1,:,:,2] = self.kwc[m-1]
                x_train[m-1,:,:,3] = self.ksc[m-1]
            if self.ifeatures == 3:
                x_train[m-1,:,:,0] = self.wc[m-1]
                x_train[m-1,:,:,1] = self.sc[m-1]
                x_train[m-1,:,:,2] = self.wcx[m-1]
                x_train[m-1,:,:,3] = self.wcy[m-1]
                x_train[m-1,:,:,4] = self.wcxx[m-1]
                x_train[m-1,:,:,5] = self.wcyy[m-1]
                x_train[m-1,:,:,6] = self.wcxy[m-1]
                x_train[m-1,:,:,7] = self.scx[m-1]
                x_train[m-1,:,:,8] = self.scy[m-1]
                x_train[m-1,:,:,9] = self.scxx[m-1]
                x_train[m-1,:,:,10] = self.scyy[m-1]
                x_train[m-1,:,:,11] = self.scxy[m-1]
                    
            if self.ilabel == 1:
                y_train[m-1,:,:,0] = self.pi[m-1]
            elif self.ilabel == 2:
                y_train[m-1,:,:,0] = self.t11[m-1]
                y_train[m-1,:,:,1] = self.t12[m-1]
                y_train[m-1,:,:,2] = self.t22[m-1]
            
        
        return x_train, y_train
    
    def gen_test_data(self):
        
        if self.ifeatures == 1:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 2), dtype='double')
        elif self.ifeatures == 2:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 4), dtype='double')
        elif self.ifeatures == 3:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 12), dtype='double')
            
        if self.ilabel == 1:
            y_test = np.zeros(shape=(1, self.nx+1, self.ny+1, 1), dtype='double')
        elif self.ilabel == 2:
            y_test = np.zeros(shape=(1, self.nx+1, self.ny+1, 3), dtype='double')
            
        m = self.n_snapshots_test

        if self.ifeatures == 1:
            x_test[0,:,:,0] = self.wc[m-1]
            x_test[0,:,:,1] = self.sc[m-1]
        if self.ifeatures == 2:
            x_test[0,:,:,0] = self.wc[m-1]
            x_test[0,:,:,1] = self.sc[m-1]
            x_test[0,:,:,2] = self.kwc[m-1]
            x_test[0,:,:,3] = self.ksc[m-1]
        if self.ifeatures == 3:
            x_test[0,:,:,0] = self.wc[m-1]
            x_test[0,:,:,1] = self.sc[m-1]
            x_test[0,:,:,2] = self.wcx[m-1]
            x_test[0,:,:,3] = self.wcy[m-1]
            x_test[0,:,:,4] = self.wcxx[m-1]
            x_test[0,:,:,5] = self.wcyy[m-1]
            x_test[0,:,:,6] = self.wcxy[m-1]
            x_test[0,:,:,7] = self.scx[m-1]
            x_test[0,:,:,8] = self.scy[m-1]
            x_test[0,:,:,9] = self.scxx[m-1]
            x_test[0,:,:,10] = self.scyy[m-1]
            x_test[0,:,:,11] = self.scxy[m-1]
        
        if self.ilabel == 1:
            y_test[0,:,:,0] = self.pi[m-1]
        elif self.ilabel == 2:
            y_test[0,:,:,0] = self.t11[m-1]
            y_test[0,:,:,1] = self.t12[m-1]
            y_test[0,:,:,2] = self.t22[m-1]            
            
        return x_test, y_test
    
#%%
#A Convolutional Neural Network class
class CNN:
    def __init__(self,x_train_f,x_train_k,y_train,nx,ny,ncf,nck,nco):
        
        '''
        initialize the CNN class
        
        Inputs
        ------
        ue : output label of the CNN model
        f : input features of the CNN model
        nx,ny : dimension of the snapshot
        nci : number of input features
        nco : number of output labels
        '''
        
        self.x_train_f = x_train_f
        self.x_train_k = x_train_k
        self.y_train = y_train
        self.nx = nx
        self.ny = ny
        self.ncf = ncf
        self.nck = nck
        self.nco = nco
        # self.model = self.CNN()
        self.model = self.CNN_PGML()
    
    def coeff_determination(self,y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
    def CNN(self):
        
        '''
        define CNN model
        
        Inputs
        ------
        ue : output labels
        f : input features (snapshot images)
        nx,ny : snapshot images shape
        nci: number of input features
        nco: number of labels
        
        Output
        ------
        model: CNN model with defined activation function, number of layers
        '''
        
        model = Sequential()
        input_img = Input(shape=(self.nx,self.ny,self.ncf))
        
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        encoded = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(encoded)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        decoded = Conv2D(nco, (4, 4), activation='linear', padding='same')(x)
        
        model = Model(input_img, decoded)
        return model
    
    def CNN_PGML(self):
        
        '''
        define CNN model
        
        Inputs
        ------
        ue : output labels
        f : input features (snapshot images)
        nx,ny : snapshot images shape
        nci: number of input features
        nco: number of labels
        
        Output
        ------
        model: CNN model with defined activation function, number of layers
        '''
               
        field = Input(shape=(self.nx,self.ny,self.ncf))
        kernels = Input(shape=(self.nx,self.ny,self.nck))
        
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(field)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        
        x = concatenate(inputs=[x, kernels])
        
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        sgs = Conv2D(nco, (4, 4), activation='linear', padding='same')(x)
        
        model = Model(inputs=[field, kernels], outputs=sgs)
        return model

    def CNN_compile(self,optimizer):
        
        '''
        compile the CNN model
        
        Inputs
        ------
        optimizer: optimizer of the CNN

        '''
        
        self.model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=[self.coeff_determination])
        
    def CNN_train(self,epochs,batch_size):
        
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
        
        history_callback = self.model.fit(x = [self.x_train_f,self.x_train_k], 
                                          y = self.y_train,
                                          epochs=epochs,batch_size=batch_size, 
                                          validation_split= 0.2,)
        return history_callback
    
    def CNN_history(self, history_callback):
        
        '''
        get the training and validation loss history
        
        Inputs
        ------
        history_callback: loss history of CNN model training
        
        Output
        ------
        loss: training loss history of CNN model training
        val_loss: validation loss history of CNN model training
        '''
        
        loss = history_callback.history["loss"]
        val_loss = history_callback.history["val_loss"]
        mse = history_callback.history['coeff_determination']
        val_mse = history_callback.history['val_coeff_determination']
               
        return loss, val_loss, mse, val_mse
            
    def CNN_predict(self,x_test_sc_f,x_test_sc_k):
        
        '''
        predict the label for input features
        
        Inputs
        ------
        ftest: test data (has same shape as input features used for training)
        
        Output
        ------
        y_predict: predicted output by the CNN (has same shape as label used for training)
        '''

        testing_time_init1 = tm.time()
        y_test = self.model.predict(x=[x_test_sc_f,x_test_sc_k])
        t1 = tm.time() - testing_time_init1
        
        testing_time_init2 = tm.time()
        y_test = self.model.predict(x=[x_test_sc_f,x_test_sc_k])
        #y_test = custom_model.predict(x_test)
        t2 = tm.time() - testing_time_init2
        
        testing_time_init3 = tm.time()
        y_test = self.model.predict(x=[x_test_sc_f,x_test_sc_k])
        y_test = self.model.predict(x=[x_test_sc_f,x_test_sc_k])
        t3 = tm.time() - testing_time_init3
        
        return y_test,t1,t2,t3
    
    def CNN_predict1(self,ftest,ist,ift,nsm):
        
        '''
        predict the label for input features
        
        Inputs
        ------
        ftest: test data (has same shape as input features used for training)
        
        Output
        ------
        y_predict: predicted output by the CNN (has same shape as label used for training)
        '''
        
        filepath = 'tcfd_paper_data/new_data/cnn_'+str(ist)+'_'+str(ift)+'_'+str(nsm)
        
        custom_model = load_model(filepath+'/CNN_model.hd5',
                                  custom_objects={'coeff_determination': self.coeff_determination})
                                  
        
        testing_time_init1 = tm.time()
        y_test = custom_model.predict(ftest)
        t1 = tm.time() - testing_time_init1
        
        testing_time_init2 = tm.time()
        y_test = custom_model.predict(ftest)
        t2 = tm.time() - testing_time_init2
        
        testing_time_init3 = tm.time()
        y_test1 = custom_model.predict(ftest)
        y_test2 = custom_model.predict(ftest)
        t3 = tm.time() - testing_time_init3
        
        return y_test,t1,t2,t3
    
    def CNN_info(self):
        
        '''
        print the CNN model summary
        '''
        
        self.model.summary()
        plot_model(self.model, to_file='cnn_model.png', show_shapes=True)
        
    def CNN_save(self,model_name):
        
        '''
        save the learned parameters (weights and bias)
        
        Inputs
        ------
        model_name: name of the file to be saved (.hd5 file)
        '''
        self.model.save(model_name)
        
#%%
# generate training and testing data for CNN
l1 = []
with open('cnn.txt') as f:
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
re = np.float64(l1[9][0]) 

obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,re=re,freq=freq,n_snapshots=n_snapshots,n_snapshots_train=n_snapshots_train, 
           n_snapshots_test=n_snapshots_test,istencil=istencil,ifeatures=ifeatures,ilabel=ilabel)

max_min = obj.max_min

x_train_sc,y_train_sc = obj.x_train,obj.y_train
x_test_sc,y_test_sc = obj.x_test,obj.y_test

x_train_sc_f = x_train_sc[:,:,:,:2]
x_train_sc_k = x_train_sc[:,:,:,2:]
x_test_sc_f = x_test_sc[:,:,:,:2]
x_test_sc_k = x_test_sc[:,:,:,2:]

nt, nx_train, ny_train, ncf = x_train_sc_f.shape
_, _, _, nck = x_train_sc_k.shape
_, _, _, nco = y_train_sc.shape 

#%%
# train the CNN model and predict for the test data
model = CNN(x_train_sc_f,x_train_sc_k,y_train_sc,nx_train,ny_train,ncf,nck,nco)
model.CNN_info()
model.CNN_compile(optimizer='adam')

#%%
training_time_init = tm.time()
history_callback = model.CNN_train(epochs=800,batch_size=32)
total_training_time = tm.time() - training_time_init

loss, val_loss, mse, val_mse = model.CNN_history(history_callback)

#%%
directory = f'nn_history/TF1/'
if not os.path.exists(directory):
    os.makedirs(directory)
    
nn_history(loss, val_loss, mse, val_mse, istencil, ifeatures, n_snapshots_train, directory)

#%%
filename = os.path.join(directory, f'CNN_model_{ifeatures}.hd5')    
model.CNN_save(filename)
filename = os.path.join(directory, f'scaling.npy')
np.save(filename,max_min)

#testing_time_init = tm.time()
y_pred_sc, t1, t2, t3 = model.CNN_predict(x_test_sc_f,x_test_sc_k)

#total_testing_time = tm.time() - testing_time_init

filename = os.path.join(directory, 'cpu_time.csv') 
with open(filename, 'a', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(['CNN',istencil, ifeatures, n_snapshots_train, total_training_time, t1, t2, t3])

#%% unscale the predicted data
y_test = np.zeros(shape=(1, nx+1, ny+1, 3), dtype='double')
y_pred = np.zeros(shape=(1, nx+1, ny+1, 3), dtype='double')

#%%
if ilabel == 1:
    for i in range(1):
        y_pred[0,:,:,i] = 0.5*(y_pred_sc[0,:,:,i]*(max_min[-1,0] - max_min[-1,1]) + (max_min[-1,0] + max_min[-1,1]))
        y_test[0,:,:,i] = 0.5*(y_test_sc[0,:,:,i]*(max_min[-1,0] - max_min[-1,1]) + (max_min[-1,0] + max_min[-1,1]))    

elif ilabel == 2:
    for i in range(3):
        y_pred[0,:,:,i] = 0.5*(y_pred_sc[0,:,:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))
        y_test[0,:,:,i] = 0.5*(y_test_sc[0,:,:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))   

nn = 2        
export_results(y_test[0], y_pred[0],  ilabel, istencil, ifeatures, n_snapshots_train, nxf, nx, nn, directory)


#%%
if ilabel == 1:
    num_bins = 64
    
    fig, axs = plt.subplots(1,1,figsize=(6,4))
    axs.set_yscale('log')

    
    # the histogram of the data
    axs.hist(y_test[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="True")
        
    axs.hist(y_pred[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="CNN")  
    
    x_ticks = np.arange(-4.1*np.std(y_test[0,:,:,0]), 4.1*np.std(y_test[0,:,:,0]), np.std(y_test[0,:,:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs.set_xlabel(r"$\tau_{11}$")
    axs.set_ylabel("PDF")
    axs.set_xticks(x_ticks)                                                           
    axs.set_xticklabels(x_labels)              
           
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["True",  "CNN"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()
    
    filename = os.path.join(directory, f'ts_cnn_{istencil}_{ifeatures}_{n_snapshots_train}.png')    
    fig.savefig(filename, bbox_inches = 'tight', dpi=200)
    
    
elif ilabel == 2:
    num_bins = 64
    
    fig, axs = plt.subplots(1,2,figsize=(6,5))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    
    # the histogram of the data
    axs[0].hist(y_test[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="True")
    
    axs[0].hist(t11s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label=r"Dynamic")
    
    axs[0].hist(y_pred[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="CNN")
    
    #axs[0].hist(t11st.flatten(), num_bins, histtype='step', alpha=1,color='k',zorder=10,
    #            linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label=r"$C_s=0.18$")
    
    
    x_ticks = np.arange(-4*np.std(y_test[0,:,:,0]), 4.1*np.std(y_test[0,:,:,0]), np.std(y_test[0,:,:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[0].set_xlabel(r"$\tau_{11}$")
    axs[0].set_ylabel("PDF")
    axs[0].set_xticks(x_ticks)                                                           
    axs[0].set_xticklabels(x_labels)              
    
    #------#
    axs[1].hist(y_test[0,:,:,1].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label="True")
    
    axs[1].hist(t12s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label=r"Dynamic")
    
    axs[1].hist(y_pred[0,:,:,1].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label="CNN")
    
    #axs[1].hist(t12st.flatten(), num_bins, histtype='step', alpha=1,color='k',zorder=10,
    #            linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label=r"$C_s=0.18$")
    
    x_ticks = np.arange(-4*np.std(y_test[0,:,:,1]), 4.1*np.std(y_test[0,:,:,1]), np.std(y_test[0,:,:,1]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[1].set_xlabel(r"$\tau_{12}$")
    #axs[1].set_ylabel("PDF")
    axs[1].set_xticks(x_ticks)                                                           
    axs[1].set_xticklabels(x_labels)              
    
    #------#
    axs[2].hist(y_test[0,:,:,2].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label="True")
    
    axs[2].hist(t22s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label=r"Dynamic")
    
    axs[2].hist(y_pred[0,:,:,2].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label="CNN")
    
    #axs[2].hist(t22st.flatten(), num_bins, histtype='step', alpha=1,color='k',zorder=10,
    #            linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label=r"$C_s=0.18$")
    
    x_ticks = np.arange(-4*np.std(y_test[0,:,:,2]), 4.1*np.std(y_test[0,:,:,2]), np.std(y_test[0,:,:,2]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[2].set_xlabel(r"$\tau_{22}$")
    #axs[2].set_ylabel("PDF")
    axs[2].set_xticks(x_ticks)                                                           
    axs[2].set_xticklabels(x_labels)              
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["True", "DSM", "CNN"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()
    
    filename = os.path.join(directory, f'ts_cnn_{istencil}_{ifeatures}_{n_snapshots_train}.png')    
    fig.savefig(filename, bbox_inches = 'tight', dpi=200)


#%%
# contour plot of shear stresses
fig, axs = plt.subplots(1,2,sharey=True,figsize=(11,5))

cbarticks = np.arange(-50,60,10)

cs = axs[0].contourf(y_test[0,:,:,0].T, cbarticks, cmap = 'jet',  )
axs[0].text(0.4, -0.1, 'True', transform=axs[0].transAxes, fontsize=14, va='top')

cs = axs[1].contourf(y_pred[0,:,:,0].T, cbarticks, cmap = 'jet', )
axs[1].text(0.4, -0.1, 'CNN', transform=axs[1].transAxes, fontsize=14, va='top')
fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)


cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, ticks=cbarticks, orientation='horizontal')
plt.show()

filename = filename = os.path.join(directory, f'contour_{istencil}_{ifeatures}_{n_snapshots_train}.png')    
fig.savefig(filename, bbox_inches = 'tight', dpi=200)

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
#%%
# nx = 128
# ny = 128
# n_field = 2
# n_kernels = 2
# nco = 1

# model = Sequential()

# field = Input(shape=(nx,ny,n_field))
# kernels = Input(shape=(nx,ny,n_kernels))

# x = Conv2D(16, (4, 4), activation='relu', padding='same')(field)
# x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
# x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
# x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)

# x = concatenate(inputs=[x, kernels])

# x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
# x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
# sgs = Conv2D(nco, (4, 4), activation='linear', padding='same')(x)

# model = Model(inputs=[field, kernels], outputs=sgs)