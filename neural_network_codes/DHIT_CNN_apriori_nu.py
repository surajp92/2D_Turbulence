# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import plot_model
from keras import backend as K
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from utils import *
import os
import time as tm
import csv

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
        
        self.max_min =  np.zeros(shape=(14, 2), dtype='double')
        
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
        
        
        self.scale_data()
        
        self.x_train,self.y_train = self.gen_train_data()
        
        self.x_test,self.y_test = self.gen_test_data()
    
    def scale_data(self):
        '''
        scaling the data between (-1,1) using (2x-(xmax+xmin))/(xmax-xmin)

        '''
        self.max_min[0,0], self.max_min[0,1] = np.max(self.uc), np.min(self.uc)
        self.max_min[1,0], self.max_min[1,1] = np.max(self.vc), np.min(self.vc)
        self.max_min[2,0], self.max_min[2,1] = np.max(self.ucx), np.min(self.ucx)
        self.max_min[3,0], self.max_min[3,1] = np.max(self.ucy), np.min(self.ucy)
        self.max_min[4,0], self.max_min[4,1] = np.max(self.vcx), np.min(self.vcx)
        self.max_min[5,0], self.max_min[5,1] = np.max(self.vcy), np.min(self.vcy)
        self.max_min[6,0], self.max_min[6,1] = np.max(self.ucxx), np.min(self.ucxx)
        self.max_min[7,0], self.max_min[7,1] = np.max(self.ucyy), np.min(self.ucyy)
        self.max_min[8,0], self.max_min[8,1] = np.max(self.vcxx), np.min(self.vcxx)
        self.max_min[9,0], self.max_min[9,1] = np.max(self.vcyy), np.min(self.vcyy)
        self.max_min[10,0], self.max_min[10,1] = np.max(self.t11), np.min(self.t11)
        self.max_min[11,0], self.max_min[11,1] = np.max(self.t12), np.min(self.t12)
        self.max_min[12,0], self.max_min[12,1] = np.max(self.t22), np.min(self.t22)
        self.max_min[13,0], self.max_min[13,1] = np.max(self.nu), np.min(self.nu)
        
        self.uc = (2.0*self.uc - (np.max(self.uc) + np.min(self.uc)))/(np.max(self.uc) - np.min(self.uc))
        self.vc = (2.0*self.vc - (np.max(self.vc) + np.min(self.vc)))/(np.max(self.vc) - np.min(self.vc))
        self.ucx = (2.0*self.ucx - (np.max(self.ucx) + np.min(self.ucx)))/(np.max(self.ucx) - np.min(self.ucx))
        self.ucy = (2.0*self.ucy - (np.max(self.ucy) + np.min(self.ucy)))/(np.max(self.ucy) - np.min(self.ucy))
        self.vcx = (2.0*self.vcx - (np.max(self.vcx) + np.min(self.vcx)))/(np.max(self.vcx) - np.min(self.vcx))
        self.vcy = (2.0*self.vcy - (np.max(self.vcy) + np.min(self.vcy)))/(np.max(self.vcy) - np.min(self.vcy))
        self.ucxx = (2.0*self.ucxx - (np.max(self.ucxx) + np.min(self.ucxx)))/(np.max(self.ucxx) - np.min(self.ucxx))
        self.ucyy = (2.0*self.ucyy - (np.max(self.ucyy) + np.min(self.ucyy)))/(np.max(self.ucyy) - np.min(self.ucyy))
        self.vcxx = (2.0*self.vcxx - (np.max(self.vcxx) + np.min(self.vcxx)))/(np.max(self.vcxx) - np.min(self.vcxx))
        self.vcyy = (2.0*self.vcyy - (np.max(self.vcyy) + np.min(self.vcyy)))/(np.max(self.vcyy) - np.min(self.vcyy))
        
        self.t11 = (2.0*self.t11 - (np.max(self.t11) + np.min(self.t11)))/(np.max(self.t11) - np.min(self.t11))
        self.t12 = (2.0*self.t12 - (np.max(self.t12) + np.min(self.t12)))/(np.max(self.t12) - np.min(self.t12))
        self.t22 = (2.0*self.t22 - (np.max(self.t22) + np.min(self.t22)))/(np.max(self.t22) - np.min(self.t22))
        
        self.nu = (2.0*self.nu - (np.max(self.nu) + np.min(self.nu)))/(np.max(self.nu) - np.min(self.nu))
        
    
    def gen_train_data(self):
        
        '''
        data generation for training and testing CNN model

        '''
        
        if self.ifeatures == 10:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 10), dtype='double')
        elif self.ifeatures == 6:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 6), dtype='double')
        elif self.ifeatures == 2:
            x_train  = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 2), dtype='double')
            
        if self.ilabel == 1:
            y_train = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 3), dtype='double')
        elif self.ilabel == 2:
            y_train = np.zeros(shape=(self.n_snapshots_train, self.nx+1, self.ny+1, 1), dtype='double')
        
        for m in range(1,self.n_snapshots_train+1):
            #m = p*self.freq
            if self.ifeatures == 10:
                x_train[m-1,:,:,0] = self.uc[m-1]
                x_train[m-1,:,:,1] = self.vc[m-1]
                x_train[m-1,:,:,2] = self.ucx[m-1]
                x_train[m-1,:,:,3] = self.ucy[m-1]
                x_train[m-1,:,:,4] = self.vcx[m-1]
                x_train[m-1,:,:,5] = self.vcy[m-1]
                x_train[m-1,:,:,6] = self.ucxx[m-1]
                x_train[m-1,:,:,7] = self.ucyy[m-1]
                x_train[m-1,:,:,8] = self.vcxx[m-1]
                x_train[m-1,:,:,9] = self.vcyy[m-1]
            if self.ifeatures == 6:
                x_train[m-1,:,:,0] = self.uc[m-1]
                x_train[m-1,:,:,1] = self.vc[m-1]
                x_train[m-1,:,:,2] = self.ucx[m-1]
                x_train[m-1,:,:,3] = self.ucy[m-1]
                x_train[m-1,:,:,4] = self.vcx[m-1]
                x_train[m-1,:,:,5] = self.vcy[m-1]
            if self.ifeatures == 2:
                x_train[m-1,:,:,0] = self.uc[m-1]
                x_train[m-1,:,:,1] = self.vc[m-1]
            
            if self.ilabel == 1:
                y_train[m-1,:,:,0] = self.t11[m-1]
                y_train[m-1,:,:,1] = self.t12[m-1]
                y_train[m-1,:,:,2] = self.t22[m-1]
            elif self.ilabel == 2:
                y_train[m-1,:,:,0] = self.nu[m-1]
        
        return x_train, y_train
    
    def gen_test_data(self):
        
        if self.ifeatures == 10:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 10), dtype='double')
        if self.ifeatures == 6:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 6), dtype='double')
        elif self.ifeatures == 2:
            x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 2), dtype='double')
            
        if self.ilabel == 1:
            y_test = np.zeros(shape=(1, self.nx+1, self.ny+1, 3), dtype='double')
        elif self.ilabel == 2:
            y_test = np.zeros(shape=(1, self.nx+1, self.ny+1, 1), dtype='double')
            
        m = self.n_snapshots_test

        if self.ifeatures == 10:
            x_test[0,:,:,0] = self.uc[m-1]
            x_test[0,:,:,1] = self.vc[m-1]
            x_test[0,:,:,2] = self.ucx[m-1]
            x_test[0,:,:,3] = self.ucy[m-1]
            x_test[0,:,:,4] = self.vcx[m-1]
            x_test[0,:,:,5] = self.vcy[m-1]
            x_test[0,:,:,6] = self.ucxx[m-1]
            x_test[0,:,:,7] = self.ucyy[m-1]
            x_test[0,:,:,8] = self.vcxx[m-1]
            x_test[0,:,:,9] = self.vcyy[m-1]
        if self.ifeatures == 6:
            x_test[0,:,:,0] = self.uc[m-1]
            x_test[0,:,:,1] = self.vc[m-1]
            x_test[0,:,:,2] = self.ucx[m-1]
            x_test[0,:,:,3] = self.ucy[m-1]
            x_test[0,:,:,4] = self.vcx[m-1]
            x_test[0,:,:,5] = self.vcy[m-1]
        if self.ifeatures == 2:
            x_test[0,:,:,0] = self.uc[m-1]
            x_test[0,:,:,1] = self.vc[m-1]
        
        if self.ilabel == 1:
            y_test[0,:,:,0] = self.t11[m-1]
            y_test[0,:,:,1] = self.t12[m-1]
            y_test[0,:,:,2] = self.t22[m-1]
        elif self.ilabel == 2:
            y_test[0,:,:,0] = self.nu[m-1]
            
        return x_test, y_test
    
#%%
#A Convolutional Neural Network class
class CNN:
    def __init__(self,x_train,y_train,nx,ny,nci,nco):
        
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
        
        self.x_train = x_train
        self.y_train = y_train
        self.nx = nx
        self.ny = ny
        self.nci = nci
        self.nco = nco
        self.model = self.CNN(x_train,y_train,nx,ny,nci,nco)
    
    def coeff_determination(self,y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
    def CNN(self,x_train,y_train,nx,ny,nci,nco):
        
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
        input_img = Input(shape=(self.nx,self.ny,self.nci))
        
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        decoded = Conv2D(nco, (3, 3), activation='linear', padding='same')(x)
        
#        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#        x = MaxPooling2D((2, 2), padding='same')(x)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        x = MaxPooling2D((2, 2), padding='same')(x)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        encoded = MaxPooling2D((2, 2), padding='same')(x)
#        
#        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#        x = UpSampling2D((2, 2))(x)
#        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#        x = UpSampling2D((2, 2))(x)
#        x = Conv2D(16, (3, 3), activation='relu')(x)
#        x = UpSampling2D((2,2))(x)
#        decoded = Conv2D(nco, (3, 3), activation='linear', padding='same')(x)
        
        model = Model(input_img, decoded)
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
        
        history_callback = self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size, 
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
            
    def CNN_predict(self,ftest):
        
        '''
        predict the label for input features
        
        Inputs
        ------
        ftest: test data (has same shape as input features used for training)
        
        Output
        ------
        y_predict: predicted output by the CNN (has same shape as label used for training)
        '''
        
        y_predict = self.model.predict(ftest)
        return y_predict
    
    def CNN_info(self):
        
        '''
        print the CNN model summary
        '''
        
        self.model.summary()
        plot_model(self.model, to_file='cnn_model.png')
        
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

#%%
obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,freq=freq,n_snapshots=n_snapshots,n_snapshots_train=n_snapshots_train, 
           n_snapshots_test=n_snapshots_test,istencil=istencil,ifeatures=ifeatures,ilabel=ilabel)

max_min = obj.max_min

#%%
x_train_sc,y_train_sc = obj.x_train,obj.y_train
x_test_sc,y_test_sc = obj.x_test,obj.y_test

nt,nx_train,ny_train,nci = x_train_sc.shape
nt,nx_train,ny_train,nco = y_train_sc.shape 

#%%
# train the CNN model and predict for the test data
model = CNN(x_train_sc,y_train_sc,nx_train,ny_train,nci,nco)
model.CNN_info()
model.CNN_compile(optimizer='adam')

#%%
training_time_init = tm.time()
history_callback = model.CNN_train(epochs=500,batch_size=32)
total_training_time = tm.time() - training_time_init

loss, val_loss, mse, val_mse = model.CNN_history(history_callback)

nn_history(loss, val_loss, mse, val_mse, istencil, ifeatures, n_snapshots_train)

#%%
model.CNN_save('./nn_history/CNN_model.h5')

testing_time_init = tm.time()
y_pred_sc = model.CNN_predict(x_test_sc)
total_testing_time = tm.time() - testing_time_init

with open('cpu_time.csv', 'a', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(['CNN',istencil, ifeatures, n_snapshots_train, total_training_time, total_testing_time])

#%% unscale the predicted data
y_test = np.zeros(shape=(1, nx+1, ny+1, 3), dtype='double')
y_pred = np.zeros(shape=(1, nx+1, ny+1, 3), dtype='double')

#%%
if ilabel == 1:
    for i in range(3):
        y_pred[0,:,:,i] = 0.5*(y_pred_sc[0,:,:,i]*(max_min[i+10,0] - max_min[i+10,1]) + (max_min[i+10,0] + max_min[i+10,1]))
        y_test[0,:,:,i] = 0.5*(y_test_sc[0,:,:,i]*(max_min[i+10,0] - max_min[i+10,1]) + (max_min[i+10,0] + max_min[i+10,1]))    

elif ilabel == 2:
    for i in range(1):
        y_pred[0,:,:,i] = 0.5*(y_pred_sc[0,:,:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))
        y_test[0,:,:,i] = 0.5*(y_test_sc[0,:,:,i]*(max_min[i+13,0] - max_min[i+13,1]) + (max_min[i+13,0] + max_min[i+13,1]))   
        
export_resutls(y_test[0], y_pred[0],  ilabel, istencil, ifeatures, n_snapshots_train, nxf, nx, nn=2)

#%%
folder = "data_"+ str(nxf) + "_" + str(nx) 
m = n_snapshots_test
file_input = "../data_spectral/"+folder+"/smag_shear_stress/ts_"+str(m)+".csv"
ts = np.genfromtxt(file_input, delimiter=',')
ts = ts.reshape((3,nx+1,ny+1))
t11s = ts[0,:,:]
t12s = ts[1,:,:]
t22s = ts[2,:,:]

folder = "data_"+ str(nxf) + "_" + str(nx) 
m = n_snapshots_test
file_input = "../data_spectral/"+folder+"/nu_smag/nus_"+str(m)+".csv"
tnu = np.genfromtxt(file_input, delimiter=',')

#%%
if ilabel == 1:
    num_bins = 64
    
    fig, axs = plt.subplots(1,3,figsize=(13,4))
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
    
    fig.savefig('nn_history/ts_cnn_'+str(istencil)+'_'+str(ifeatures)+'_'+str(n_snapshots_train)+'.pdf', bbox_inches = 'tight')

elif ilabel == 2:
    num_bins = 64
    
    fig, axs = plt.subplots(1,1,figsize=(6,4))
    axs.set_yscale('log')

    
    # the histogram of the data
    axs.hist(y_test[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
               linewidth=2.0,range=(0*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="True")
    
    axs.hist(tnu.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
                linewidth=2.0,range=(0*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label=r"Dynamic")
    
    axs.hist(y_pred[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
               linewidth=2.0,range=(0*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="CNN")  
    
    x_ticks = np.arange(0*np.std(y_test[0,:,:,0]), 4.1*np.std(y_test[0,:,:,0]), np.std(y_test[0,:,:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(0,5)]
    axs.set_xlabel(r"$\tau_{11}$")
    axs.set_ylabel("PDF")
    axs.set_xticks(x_ticks)                                                           
    axs.set_xticklabels(x_labels)              
           
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["True", "DSM", "CNN"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()
    
    fig.savefig('nn_history/ts_cnn_'+str(istencil)+'_'+str(ifeatures)+'_'+str(n_snapshots_train)+'.pdf', bbox_inches = 'tight')
    
#%%
# contour plot of shear stresses
fig, axs = plt.subplots(1,3,sharey=True,figsize=(10.5,3.5))

cs = axs[0].contourf(y_test[0,:,:,0].T, 120, cmap = 'jet', interpolation='bilinear')
axs[0].text(0.4, -0.1, 'True', transform=axs[0].transAxes, fontsize=14, va='top')

cs = axs[1].contourf(y_test[0,:,:,1].T, 120, cmap = 'jet', interpolation='bilinear')
axs[1].text(0.4, -0.1, 'CNN', transform=axs[1].transAxes, fontsize=14, va='top')

cs = axs[2].contourf(t11s.T, 120, cmap = 'jet', interpolation='bilinear')
axs[2].text(0.4, -0.1, 'Smag', transform=axs[2].transAxes, fontsize=14, va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()


