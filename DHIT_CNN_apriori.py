# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: Suraj Pawar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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
        
        x_train  = np.zeros(shape=(self.n_snapshots-1, self.nx+1, self.ny+1, 2), dtype='double')
        y_train = np.zeros(shape=(self.n_snapshots-1, self.nx+1, self.ny+1, 3), dtype='double')
        
        x_test  = np.zeros(shape=(1, self.nx+1, self.ny+1, 2), dtype='double')
        y_test = np.zeros(shape=(1, self.nx+1, self.ny+1, 3), dtype='double')
        
        for m in range(1,self.n_snapshots):
            folder = "data_"+ str(self.nxf) + "_" + str(self.nx) 
            
            file_input = "spectral/"+folder+"/uc/uc_"+str(self.freq*m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            x_train[m-1,:,:,0] = data_input
            
            file_input = "spectral/"+folder+"/vc/vc_"+str(self.freq*m)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            x_train[m-1,:,:,1] = data_input
            
#            file_input = "spectral/"+folder+"/uuc/uuc_"+str(self.freq*m)+".csv"
#            data_input = np.genfromtxt(file_input, delimiter=',')
#            x_train[m-1,:,:,2] = data_input
#            
#            file_input = "spectral/"+folder+"/uvc/uvc_"+str(self.freq*m)+".csv"
#            data_input = np.genfromtxt(file_input, delimiter=',')
#            x_train[m-1,:,:,3] = data_input
#            
#            file_input = "spectral/"+folder+"/vvc/vvc_"+str(self.freq*m)+".csv"
#            data_input = np.genfromtxt(file_input, delimiter=',')
#            x_train[m-1,:,:,4] = data_input
                       
            file_output = "spectral/"+folder+"/true_shear_stress/t_"+str(self.freq*m)+".csv"
#            file_output = "spectral/"+folder+"/nu_true/nut_"+str(self.freq*m)+".csv"
            data_output = np.genfromtxt(file_output, delimiter=',')
            data_output = data_output.reshape((3,self.nx+1,self.ny+1))
            y_train[m-1,:,:,0] = data_output[0,:,:] # tau_11
            y_train[m-1,:,:,1] = data_output[1,:,:] # tau_12
            y_train[m-1,:,:,2] = data_output[2,:,:] # tau_22
        
        m = (self.n_snapshots)*self.freq
        file_input = "spectral/"+folder+"/uc/uc_"+str(m)+".csv"
        data_input = np.genfromtxt(file_input, delimiter=',')
        x_test[0,:,:,0] = data_input
        
        file_input = "spectral/"+folder+"/vc/vc_"+str(m)+".csv"
        data_input = np.genfromtxt(file_input, delimiter=',')
        x_test[0,:,:,1] = data_input
        
#        file_input = "spectral/"+folder+"/uuc/uuc_"+str(m)+".csv"
#        data_input = np.genfromtxt(file_input, delimiter=',')
#        x_test[0,:,:,2] = data_input
#        
#        file_input = "spectral/"+folder+"/uvc/uvc_"+str(m)+".csv"
#        data_input = np.genfromtxt(file_input, delimiter=',')
#        x_test[0,:,:,3] = data_input
#        
#        file_input = "spectral/"+folder+"/vvc/vvc_"+str(m)+".csv"
#        data_input = np.genfromtxt(file_input, delimiter=',')
#        x_test[0,:,:,4] = data_input
                   
        file_output = "spectral/"+folder+"/true_shear_stress/t_"+str(m)+".csv"
#        file_output = "spectral/"+folder+"/nu_true/nut_"+str(m)+".csv"
        data_output = np.genfromtxt(file_output, delimiter=',')
        data_output = data_output.reshape((3,self.nx+1,self.ny+1))
        y_test[0,:,:,0] = data_output[0,:,:] # tau_11
        y_test[0,:,:,1] = data_output[0,:,:] # tau_12
        y_test[0,:,:,2] = data_output[0,:,:] # tau_22
            
        return x_train, y_train, x_test, y_test
    
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
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        decoded = Conv2D(nco, (3, 3), activation='linear', padding='same')(x)
        model = Model(input_img, decoded)
        return model

    def CNN_compile(self,optimizer):
        
        '''
        compile the CNN model
        
        Inputs
        ------
        optimizer: optimizer of the CNN

        '''
        
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        
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
                                          validation_split= 0.15,)
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
        
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.semilogy(epochs, loss, 'b', label='Training loss')
        plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        
        return loss, val_loss
            
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
freq = 1
n_snapshots = 400
nxf, nyf = 1024, 1024
nx, ny = 64, 64

#%%
obj = DHIT(nx=nx,ny=ny,nxf=nxf,nyf=nyf,freq=freq,n_snapshots=n_snapshots)

x_train,y_train = obj.x_train,obj.y_train
x_test,y_test = obj.x_test,obj.y_test

nt,nx_train,ny_train,nci = x_train.shape
nt,nx_train,ny_train,nco = y_train.shape 

#%%
# train the CNN model and predict for the test data
model = CNN(x_train,y_train,nx_train,ny_train,nci,nco)
model.CNN_info()
model.CNN_compile(optimizer='adam')

#%%
history_callback = model.CNN_train(epochs=1000,batch_size=32)
loss_history, val_loss_history = model.CNN_history(history_callback)

#%%
model.CNN_save('CNN_model.h5')
y_pred = model.CNN_predict(x_test)

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
axs[0].hist(y_test[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[0,:,:,0])),density=True,label="True")

axs[0].hist(y_pred[0,:,:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,0]),4*np.std(y_test[:,0])),density=True,label="CNN")

axs[0].hist(t11s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[0,:,:,0]), 4.1*np.std(y_test[0,:,:,0]), np.std(y_test[0,:,:,0]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[0].set_title(r"$\tau_{11}$")
axs[0].set_xticks(x_ticks)                                                           
axs[0].set_xticklabels(x_labels)              
axs[0].legend()

#------#
axs[1].hist(y_test[0,:,:,1].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label="True")

axs[1].hist(y_pred[0,:,:,1].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label="CNN")

axs[1].hist(t12s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[0,:,:,1]),4*np.std(y_test[0,:,:,1])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[0,:,:,1]), 4.1*np.std(y_test[0,:,:,1]), np.std(y_test[0,:,:,1]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[1].set_title(r"$\tau_{12}$")
axs[1].set_xticks(x_ticks)                                                           
axs[1].set_xticklabels(x_labels)              
axs[1].legend()

#------#
axs[2].hist(y_test[0,:,:,2].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label="True")

axs[2].hist(y_pred[0,:,:,2].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
           linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label="CNN")

axs[2].hist(t22s.flatten(), num_bins, histtype='step', alpha=1,color='g',zorder=10,
            linewidth=2.0,range=(-4*np.std(y_test[0,:,:,2]),4*np.std(y_test[0,:,:,2])),density=True,label=r"$C_s=0.18$")

x_ticks = np.arange(-4*np.std(y_test[0,:,:,2]), 4.1*np.std(y_test[0,:,:,2]), np.std(y_test[0,:,:,2]))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
axs[2].set_title(r"$\tau_{22}$")
axs[2].set_xticks(x_ticks)                                                           
axs[2].set_xticklabels(x_labels)              
axs[2].legend()

fig.tight_layout()
plt.show()

fig.savefig("ts_cnn.pdf", bbox_inches = 'tight')

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


