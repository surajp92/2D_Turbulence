# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from scipy.stats import norm 
from keras import backend as K

#%%
#Class of problem to solve 2D decaying homogeneous isotrpic turbulence
class DHIT:
    def __init__(self,n_snapshots,nx,ny):
        
        '''
        initialize the DHIT class
        
        Inputs
        ------
        n_snapshots : number of snapshots available
        nx,ny : dimension of the snapshot

        '''
        
        self.nx = nx
        self.ny = ny
        self.n_snapshots = n_snapshots
        self.x_train,self.y_train,self.x_test,self.y_test = self.gen_data()
        
    def gen_data(self):
        
        '''
        data generation for training and testing CNN model

        '''
        n_snapshots_test = 10
        n_snapshots_train = self.n_snapshots - n_snapshots_test 
               
        for m in range(1,n_snapshots_train):
            file_input = "spectral/Re_4000/uc/uc_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            
            nx,ny = data_input.shape
            nt = int((nx-2)*(ny-2))
            
            x_t = np.zeros((nt,9))
            y_t = np.zeros((nt,1))
            
            n = 0
            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    x_t[n,0] = data_input[i,j]
                    x_t[n,1] = data_input[i,j-1]
                    x_t[n,2] = data_input[i,j+1]
                    x_t[n,3] = data_input[i-1,j]
                    x_t[n,4] = data_input[i+1,j]
                    x_t[n,5] = data_input[i-1,j-1]
                    x_t[n,6] = data_input[i-1,j+1]
                    x_t[n,7] = data_input[i+1,j-1]
                    x_t[n,8] = data_input[i+1,j+1]
                    y_t[n,0] = data_input[i,j]
                    n = n+1
            
            if m == 1:
                x_train = x_t
                y_train = y_t
            else:
                x_train = np.vstack((x_train,x_t))
                y_train = np.vstack((y_train,y_t))
        
        for m in range(n_snapshots_train,self.n_snapshots):
            file_input = "spectral/Re_4000/uc/uc_"+str(m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            
            nx,ny = data_input.shape
            nt = int((nx-2)*(ny-2))
            
            x_t = np.zeros((nt,9))
            y_t = np.zeros((nt,1))
            
            n = 0
            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    x_t[n,0] = data_input[i,j]
                    x_t[n,1] = data_input[i,j-1]
                    x_t[n,2] = data_input[i,j+1]
                    x_t[n,3] = data_input[i-1,j]
                    x_t[n,4] = data_input[i+1,j]
                    x_t[n,5] = data_input[i-1,j-1]
                    x_t[n,6] = data_input[i-1,j+1]
                    x_t[n,7] = data_input[i+1,j-1]
                    x_t[n,8] = data_input[i+1,j+1]
                    y_t[n,0] = data_input[i,j]
                    n = n+1
            
            if m == n_snapshots_train:
                x_test = x_t
                y_test = y_t
            else:
                x_test = np.vstack((x_test,x_t))
                y_test = np.vstack((y_test,y_t))
        
        return x_train, y_train, x_test, y_test
    
#%%
#A Convolutional Neural Network class
class DNN:
    def __init__(self,x_train,y_train,nf,nl):
        
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
        
        model = Sequential()
        input_layer = Input(shape=(self.nf,))
        
        x = Dense(120, activation='relu',  use_bias=True)(input_layer)
        x = Dense(120, activation='relu',  use_bias=True)(x)
        
        output_layer = Dense(nl, activation='linear', use_bias=True)(x)
        
        model = Model(input_layer, output_layer)
        
        return model

    def DNN_compile(self,optimizer):
        
        '''
        compile the CNN model
        
        Inputs
        ------
        optimizer: optimizer of the CNN

        '''
        
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[self.coeff_determination])
        
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
                                          validation_split= 0.15,callbacks=callbacks_list)
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
        
        y_test = self.model.predict(x_test)
        return y_test
    
    def DNN_info(self):
        
        '''
        print the CNN model summary
        '''
        
        self.model.summary()
        plot_model(self.model, to_file='dnn_model.png')
     
        
#%%
# generate training and testing data for CNN
obj = DHIT(n_snapshots=50,nx=64,ny=64)

x_train,y_train = obj.x_train,obj.y_train
x_test,y_test = obj.x_test,obj.y_test

ns_train,nf = x_train.shape
ns_train,nl = y_train.shape 

indices = np.random.randint(0,x_train.shape[0],1000)
x_train = x_train[indices]
y_train = y_train[indices]

#%%
# train the CNN model and predict for the test data
model=DNN(x_train,y_train,nf,nl)
model.DNN_info()
model.DNN_compile(optimizer='adam')

#%%
history_callback = model.DNN_train(epochs=300,batch_size=32)#,model_name="dnn_best_model.hd5")

#%%
loss, val_loss = model.DNN_history(history_callback)

y_pred = model.DNN_predict(x_test)


#%%
# histogram plot for shear stresses along with probability density function 
# PDF formula: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
nx = 64
ny = 64

y11t = y_test[49,:,:,0].flatten()
mut = np.mean(y11t)
sigmat = np.std(y11t)

y11p = y_pred[49,:,:,0].flatten()
mup = np.mean(y11p)
sigmap = np.std(y11p)

ts = np.genfromtxt('spectral/Re_8000/smag_shear_stress/ts_50.csv', delimiter=',') 
ts = ts.reshape((3,nx+1,ny+1))
y11s = ts[0,:,:].flatten()
#y11s = ts[1,:,:].flatten()
#y11s = ts[2,:,:].flatten()
mus = np.mean(y11s)
sigmas = np.std(y11s)

num_bins = 64

fig, axs = plt.subplots(1,2,figsize=(10,4.5))
axs[0].set_yscale('log')

# the histogram of the data
ntrue, binst, patchest = axs[0].hist(y11t, num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*sigmat,4*sigmat),density=True,label="True")

npred, binsp, patchesp = axs[0].hist(y11p, num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,range=(-4*sigmat,4*sigmat),density=True,label="CNN")

nsmag, binss, patchess = axs[0].hist(y11s, num_bins, histtype='step', alpha=1,color='g',zorder=10,
                                 linewidth=2.0,range=(-4*sigmat,4*sigmat),density=True,label="Smag")

x_ticks = np.arange(-4*sigmat, 4.1*sigmat, sigmat)                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]


axs[0].set_title(r"$\tau_{22}$")
axs[0].set_xticks(x_ticks)                                                           
axs[0].set_xticklabels(x_labels)              

# Tweak spacing to prevent clipping of ylabel
axs[0].legend()

x_plott = np.linspace(min(y11t), max(y11t), 1000)
x_plotp = np.linspace(min(y11p), max(y11p), 1000)
x_plots = np.linspace(min(y11s), max(y11s), 1000)

axs[1].plot(x_plott, norm.pdf(x_plott, mut, sigmat), 'r-', lw=3, label="True")
axs[1].plot(x_plotp, norm.pdf(x_plotp, mup, sigmap), 'b-', lw=3, label="CNN")
axs[1].plot(x_plots, norm.pdf(x_plots, mus, sigmas), 'g-', lw=3, label="Smag")       
                                                    
axs[1].legend(loc='best')

axs[1].set_xlim(-4*sigmat,4*sigmat)  
axs[1].set_title(r"$\tau_{22}$")                     
axs[1].set_xticks(x_ticks)                                                           
axs[1].set_xticklabels(x_labels)              

fig.tight_layout()
plt.show()

fig.savefig("extrapolation_t11.pdf", bbox_inches = 'tight')

#%%
# contour plot of shear stresses
fig, axs = plt.subplots(1,3,sharey=True,figsize=(10.5,3.5))

cs = axs[0].contourf(y_test[4,:,:,0].T, 120, cmap = 'jet', interpolation='bilinear')
axs[0].text(0.4, -0.1, 'True', transform=axs[0].transAxes, fontsize=14, va='top')

cs = axs[1].contourf(y_test[4,:,:,0].T, 120, cmap = 'jet', interpolation='bilinear')
axs[1].text(0.4, -0.1, 'CNN', transform=axs[1].transAxes, fontsize=14, va='top')

cs = axs[2].contourf(ts[0,:,:].T, 120, cmap = 'jet', interpolation='bilinear')
axs[2].text(0.4, -0.1, 'Smag', transform=axs[2].transAxes, fontsize=14, va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

