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
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

#%%
#Class of problem to solve 2D decaying homogeneous isotrpic turbulence
class DHIT:
    def __init__(self,nx,ny,nxf,nyf,freq,n_snapshots):
        
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
        self.x_train,self.y_train,self.x_test,self.y_test = self.gen_data()
        
    def gen_data(self):
        
        '''
        data generation for training and testing CNN model

        '''
        
        for m in range(1,self.n_snapshots):
            folder = "data_"+ str(self.nxf) + "_" + str(self.nx) 
#            file_input = "spectral/"+folder+"/00_wc/wc_"+str(m*self.freq)+".csv"
#            uc = np.genfromtxt(file_input, delimiter=',')
#            file_input = "spectral/"+folder+"/00_sc/sc_"+str(m*self.freq)+".csv"
#            vc = np.genfromtxt(file_input, delimiter=',')
#            file_input = "spectral/"+folder+"/00_sgs/sgs_"+str(m*self.freq)+".csv"
#            sgs = np.genfromtxt(file_input, delimiter=',')
            file_input = "spectral/"+folder+"/uc/uc_"+str(m*self.freq)+".csv"
            uc = np.genfromtxt(file_input, delimiter=',')
            file_input = "spectral/"+folder+"/vc/vc_"+str(m*self.freq)+".csv"
            vc = np.genfromtxt(file_input, delimiter=',')
            file_input = "spectral/"+folder+"/Sc/Sc_"+str(m*self.freq)+".csv"
            S = np.genfromtxt(file_input, delimiter=',')
            file_input = "spectral/"+folder+"/nu_true/nut_"+str(m*self.freq)+".csv"
            nu = np.genfromtxt(file_input, delimiter=',')
            nu = nu.reshape((3,self.nx+1,self.ny+1))
            nu11 = nu[0,:,:]
            nu12 = nu[1,:,:]
            nu22 = nu[2,:,:]
            file_input = "spectral/"+folder+"/true_shear_stress/t_"+str(m*self.freq)+".csv"
            t = np.genfromtxt(file_input, delimiter=',')
            t = t.reshape((3,self.nx+1,self.ny+1))
            t11 = t[0,:,:]
            t12 = t[1,:,:]
            t22 = t[2,:,:]
            
            nx,ny = uc.shape
            nt = int((nx-2)*(ny-2))
            
            x_t = np.zeros((nt,19))
            y_t = np.zeros((nt,3))
            
            n = 0
            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    x_t[n,0],x_t[n,9] = uc[i,j], vc[i,j]
                    x_t[n,1],x_t[n,10] = uc[i,j-1], vc[i,j-1]
                    x_t[n,2],x_t[n,11] = uc[i,j+1], vc[i,j+1]
                    x_t[n,3],x_t[n,12] = uc[i-1,j], vc[i-1,j]
                    x_t[n,4],x_t[n,13] = uc[i+1,j], vc[i+1,j]
                    x_t[n,5],x_t[n,14] = uc[i-1,j-1], vc[i-1,j-1]
                    x_t[n,6],x_t[n,15] = uc[i-1,j+1], vc[i-1,j+1]
                    x_t[n,7],x_t[n,16] = uc[i+1,j-1], vc[i+1,j-1]
                    x_t[n,8],x_t[n,17] = uc[i+1,j+1], vc[i+1,j+1]
                    x_t[n,18] = S[i,j]
                    
                    y_t[n,0] = t11[i,j]
                    y_t[n,1] = t12[i,j]
                    y_t[n,2] = t22[i,j]
                    n = n+1
            
            if m == 1:
                x_train = x_t
                y_train = y_t
            else:
                x_train = np.vstack((x_train,x_t))
                y_train = np.vstack((y_train,y_t))
        
        m = (self.n_snapshots)*self.freq
#        file_input = "spectral/"+folder+"/00_wc/wc_"+str(m)+".csv"
#        uc = np.genfromtxt(file_input, delimiter=',')
#        file_input = "spectral/"+folder+"/00_sc/sc_"+str(m)+".csv"
#        vc = np.genfromtxt(file_input, delimiter=',')
#        file_input = "spectral/"+folder+"/00_sgs/sgs_"+str(m)+".csv"
#        sgs = np.genfromtxt(file_input, delimiter=',')
        file_input = "spectral/"+folder+"/uc/uc_"+str(m)+".csv"
        uc = np.genfromtxt(file_input, delimiter=',')
        file_input = "spectral/"+folder+"/vc/vc_"+str(m)+".csv"
        vc = np.genfromtxt(file_input, delimiter=',')
        file_input = "spectral/"+folder+"/Sc/Sc_"+str(m)+".csv"
        S = np.genfromtxt(file_input, delimiter=',')
        file_input = "spectral/"+folder+"/nu_true/nut_"+str(m)+".csv"
        nu = np.genfromtxt(file_input, delimiter=',')
        nu = nu.reshape((3,self.nx+1,self.ny+1))
        nu11 = nu[0,:,:]
        nu12 = nu[1,:,:]
        nu22 = nu[2,:,:]
        file_input = "spectral/"+folder+"/true_shear_stress/t_"+str(m)+".csv"
        t = np.genfromtxt(file_input, delimiter=',')
        t = t.reshape((3,self.nx+1,self.ny+1))
        t11 = t[0,:,:]
        t12 = t[1,:,:]
        t22 = t[2,:,:]
        
        nx,ny = uc.shape
        nt = int((nx-2)*(ny-2))
        
        x_t = np.zeros((nt,19))
        y_t = np.zeros((nt,3))
        
        n = 0
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                x_t[n,0],x_t[n,9] = uc[i,j], vc[i,j]
                x_t[n,1],x_t[n,10] = uc[i,j-1], vc[i,j-1]
                x_t[n,2],x_t[n,11] = uc[i,j+1], vc[i,j+1]
                x_t[n,3],x_t[n,12] = uc[i-1,j], vc[i-1,j]
                x_t[n,4],x_t[n,13] = uc[i+1,j], vc[i+1,j]
                x_t[n,5],x_t[n,14] = uc[i-1,j-1], vc[i-1,j-1]
                x_t[n,6],x_t[n,15] = uc[i-1,j+1], vc[i-1,j+1]
                x_t[n,7],x_t[n,16] = uc[i+1,j-1], vc[i+1,j-1]
                x_t[n,8],x_t[n,17] = uc[i+1,j+1], vc[i+1,j+1]
                x_t[n,18] = S[i,j]
                
                y_t[n,0] = t11[i,j]
                y_t[n,1] = t12[i,j]
                y_t[n,2] = t22[i,j]
                n = n+1
        
        x_test = x_t
        y_test = y_t
        
        return x_train, y_train, x_test, y_test
    
#%%
#A Convolutional Neural Network class
class DNN:
    def __init__(self,x_train,y_train,x_valid,y_valid,nf,nl):
        
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
        
        x = Dense(50, activation='relu',  use_bias=True)(input_layer)
        x = Dense(50, activation='relu',  use_bias=True)(x)
        
        output_layer = Dense(nl, activation='linear', use_bias=True)(x)
        
        model = Model(input_layer, output_layer)
        
        return model

    def DNN_compile(self):
        
        '''
        compile the CNN model
        
        Inputs
        ------
        optimizer: optimizer of the CNN

        '''
        
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        
    def DNN_train(self,epochs,batch_size):
        
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
        
        filepath = "dnn_best_model.hd5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        history_callback = self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size, 
                                          validation_data= (self.x_valid,self.y_valid),callbacks=callbacks_list)
        return history_callback
    
    def DNN_history(self, history_callback):
        
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
        
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.semilogy(epochs, loss, 'b', label='Training loss')
        plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        return loss, val_loss
            
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
        
        custom_model = load_model('dnn_best_model.hd5')
                                  #custom_objects={'coeff_determination':self.coeff_determination})
        y_test = custom_model.predict(x_test)
        return y_test
    
    def DNN_info(self):
        
        '''
        print the CNN model summary
        '''
        
        self.model.summary()
        plot_model(self.model, to_file='dnn_model.png')
     
        
#%%
# generate training and testing data for CNN
freq = 5
n_snapshots = 80
nxf, nyf = 1024, 1024
nx, ny = 64, 64

obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,freq=freq,n_snapshots=n_snapshots)

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

x_test_sc = sc_input.transform(x_test)

#%%
x_train, x_valid, y_train, y_valid = train_test_split(data_sc, labels_sc, test_size=0.2, shuffle= True)

ns_train,nf = x_train.shape
ns_train,nl = y_train.shape 
#%%
# train the CNN model and predict for the test data
model=DNN(x_train,y_train,x_valid,y_valid,nf,nl)
model.DNN_info()
model.DNN_compile()

#%%
history_callback = model.DNN_train(epochs=500,batch_size=64)#,model_name="dnn_best_model.hd5")
loss, val_loss = model.DNN_history(history_callback)

#%%
#y_pred = model.DNN_predict(x_test)
y_pred_sc = model.DNN_predict(x_test_sc)
y_pred = sc_output.inverse_transform(y_pred_sc)

#%%
folder = "data_"+ str(nxf) + "_" + str(nx) 
m = n_snapshots*freq
file_input = "spectral/"+folder+"/smag_shear_stress/ts_"+str(m)+".csv"
ts = np.genfromtxt(file_input, delimiter=',')
ts = ts.reshape((3,nx+1,ny+1))
t11s = ts[0,:,:]
t12s = ts[1,:,:]
t22s = ts[2,:,:]

#%%
# histogram plot for shear stresses along with probability density function 
# PDF formula: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
num_bins = 64

fig, axs = plt.subplots(1,3,figsize=(12,4))
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

# the histogram of the data
axs[0].hist(y_test[:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label="True")

axs[0].hist(y_pred[:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label="DNN")

axs[0].hist(t11s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[:,0]), 4.1*np.std(y_test[:,0]), np.std(y_test[:,0]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[0].set_title(r"$\tau_{11}$")
axs[0].set_xticks(x_ticks)                                                           
axs[0].set_xticklabels(x_labels)              
axs[0].legend()

#------#
axs[1].hist(y_test[:,1].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label="True")

axs[1].hist(y_pred[:,1].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label="DNN")

axs[1].hist(t12s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[:,1]), 4.1*np.std(y_test[:,1]), np.std(y_test[:,1]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[1].set_title(r"$\tau_{12}$")
axs[1].set_xticks(x_ticks)                                                           
axs[1].set_xticklabels(x_labels)              
axs[1].legend()

#------#
axs[2].hist(y_test[:,2].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label="True")

axs[2].hist(y_pred[:,2].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label="DNN")

axs[2].hist(t22s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[:,2]), 4.1*np.std(y_test[:,2]), np.std(y_test[:,2]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[2].set_title(r"$\tau_{22}$")
axs[2].set_xticks(x_ticks)                                                           
axs[2].set_xticklabels(x_labels)              
axs[2].legend()

fig.tight_layout()
plt.show()

fig.savefig("ts_dnn.pdf", bbox_inches = 'tight')
