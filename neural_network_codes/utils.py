#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:28:06 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
import os

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#%%
def export_resutls(y_test, y_pred, ilabel, istencil, ifeatures, n_snapshots_train, nxf, nx, nn):
    '''
    export the test data and ML predicted results
    
    Inputs
    ------
    y_test: test data (labels)
    y_ped: ML predicted results (labels)
    ilabel: flag for label 
    
    Output
    ------
    csv files: results saved in csv format
    '''

    if nn == 1:
        folder = "nn_history/"
        if ilabel == 1:
            filename = folder+"/y_test_sgs_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_test, delimiter=",")
            
            filename = folder+"/y_pred_sgs_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_pred, delimiter=",")
        
        if ilabel == 2:
            filename = folder+"/y_test_nu_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_test, delimiter=",")
            
            filename = folder+"/y_pred_nu_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_pred, delimiter=",")
            
    elif nn == 2:
        folder = "nn_history/"
        y_t = np.zeros((y_test.shape[0]*y_test.shape[1],y_test.shape[2]))
        y_p = np.zeros((y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]))
        for i in range(3):
            y_t[:,i] = y_test[:,:,i].flatten() 
            y_p[:,i] = y_pred[:,:,i].flatten() 

        if ilabel == 1:
            filename = folder+"/y_test_sgs_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_t, delimiter=",")
            
            filename = folder+"/y_pred_sgs_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_p, delimiter=",")
        
        if ilabel == 2:
            filename = folder+"/y_test_nu_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_t[:,0], delimiter=",")
            
            filename = folder+"/y_pred_nu_"+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+".csv"
            np.savetxt(filename, y_p[:,0], delimiter=",")
        
#%%
def plot_dynamic_cs2(nxf, nx):
    folder = "data_"+ str(nxf) + "_" + str(nx) 
            
    file_input = "../data_spectral/"+folder+"/cs2/cs2.csv"
    data_input = np.genfromtxt(file_input, delimiter=',',skip_header=1)
    t = data_input[:,0]/100
    cs2 = data_input[:,2]
    
    fig, axs = plt.subplots(1, 1, figsize=(6,3))#, constrained_layout=True)
    axs.plot(t,cs2, color='k', lw = 2.5, linestyle='-', label=r'$C_s$', zorder=5)
    

    #axs.grid(True)
    axs.set_xlim([t[0],t[int(t.shape[0]-1)]])
    axs.set_ylabel('$C_s$', fontsize = 14)
    axs.set_xlabel('$t$', fontsize = 14)
    axs.legend()
    
    fig.tight_layout() 
    #fig.subplots_adjust(bottom=0.2)
    fig.savefig('cs.pdf')
    fig.savefig('cs.eps')
    

plot_dynamic_cs2(nxf=1024, nx=64)

#%%
def nn_history(loss, val_loss, mse, val_mse, istencil, ifeatures, n_snapshots_train):
    
    epochs = range(1, len(loss) + 1)
    
    history = np.zeros((len(loss), 5))
    
    history[:,0] = epochs
    history[:,1] = loss
    history[:,2] = val_loss
    history[:,3] = mse
    history[:,4] = val_mse
    
    np.savetxt('nn_history/history_'+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+'.csv', history, delimiter=",")
    
    fig, axs = plt.subplots(1,2,figsize=(9,4))
    axs[0].semilogy(epochs, loss, 'b', label='Training loss')
    axs[0].semilogy(epochs, val_loss, 'r', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()
    
    axs[1].semilogy(epochs, mse, 'b', label='Training MSE')
    axs[1].semilogy(epochs, val_mse, 'r', label='Validation MSE')
    axs[1].set_title('Training and validation MSE')
    axs[1].legend()
    
    fig.tight_layout()
    plt.show()
    fig.savefig('nn_history/loss_mse_'+str(istencil)+"_"+str(ifeatures)+"_"+str(n_snapshots_train)+'.pdf')
    
#%%
def plot_velocity_field():
    folder = "data_"+ str(1024) + "_" + str(64) 
    file_input = "../data_spectral/"+folder+"/vc/vc_"+str(350)+".csv"
    data_input = np.genfromtxt(file_input, delimiter=',')
    uc = data_input
    
    file_output = "../data_spectral/"+folder+"/true_shear_stress/t_"+str(350)+".csv"
    data_output = np.genfromtxt(file_output, delimiter=',')
    data_output = data_output.reshape((3,65,65))
    t11 = data_output[0,:,:]
    t12 = data_output[1,:,:]
    t22 = data_output[2,:,:]
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.contourf(t22.T, 120, cmap = 'jet', interpolation='bilinear',levels=np.linspace(-0.05,0.05,21), extend="both")
    axs.set_xticks([])
    axs.set_yticks([])
    
    fig.tight_layout() 
    plt.show()
    fig.savefig('t22.png',dpi=100,bbox_inches = 'tight',pad_inches = 0)

#plot_velocity_field()

#%%
def compare_pdf():
    folder = "data_"+ str(1024) + "_" + str(64) 
    file_output = "../data_spectral/"+folder+"/true_shear_stress/t_"+str(250)+".csv"
    data_output = np.genfromtxt(file_output, delimiter=',')
    data_output = data_output.reshape((3,65,65))
    y_test = np.zeros((65*65,3))
    y_test[:,0] = data_output[0,:,:].flatten()
    y_test[:,1] = data_output[1,:,:].flatten()
    y_test[:,2] = data_output[2,:,:].flatten()
    
    file_output = "../data_spectral/"+folder+"/true_shear_stress/t_"+str(400)+".csv"
    data_output = np.genfromtxt(file_output, delimiter=',')
    data_output = data_output.reshape((3,65,65))
    y_pred = np.zeros((65*65,3))
    y_pred[:,0] = data_output[0,:,:].flatten()
    y_pred[:,1] = data_output[1,:,:].flatten()
    y_pred[:,2] = data_output[2,:,:].flatten()
    
    num_bins = 64

    fig, axs = plt.subplots(1,3,figsize=(13,4))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    
    # the histogram of the data
    axs[0].hist(y_test[:,0].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label="True")
    
    axs[0].hist(y_pred[:,0].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,0]),4*np.std(y_test[:,0])),density=True,label="DNN")
        
    x_ticks = np.arange(-4*np.std(y_test[:,0]), 4.1*np.std(y_test[:,0]), np.std(y_test[:,0]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[0].set_xlabel(r"$\tau_{11}$")
    axs[0].set_ylabel("PDF")
    axs[0].set_xticks(x_ticks)                                                           
    axs[0].set_xticklabels(x_labels)              
    
    #------#
    axs[1].hist(y_test[:,1].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label="True")
    
    axs[1].hist(y_pred[:,1].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,1]),4*np.std(y_test[:,1])),density=True,label="DNN")
    
    x_ticks = np.arange(-4*np.std(y_test[:,1]), 4.1*np.std(y_test[:,1]), np.std(y_test[:,1]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[1].set_xlabel(r"$\tau_{12}$")
    axs[1].set_ylabel("PDF")
    axs[1].set_xticks(x_ticks)                                                           
    axs[1].set_xticklabels(x_labels)              
    
    #------#
    axs[2].hist(y_test[:,2].flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label="True")
    
    axs[2].hist(y_pred[:,2].flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                     linewidth=2.0,range=(-4*np.std(y_test[:,2]),4*np.std(y_test[:,2])),density=True,label="DNN")
    
    x_ticks = np.arange(-4*np.std(y_test[:,2]), 4.1*np.std(y_test[:,2]), np.std(y_test[:,2]))                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]
    axs[2].set_xlabel(r"$\tau_{22}$")
    axs[2].set_ylabel("PDF")
    axs[2].set_xticks(x_ticks)                                                           
    axs[2].set_xticklabels(x_labels)              
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.25)
    line_labels = ["250", "400"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.3, ncol=3, labelspacing=0.,  prop={'size': 13} )
    plt.show()          

#compare_pdf()
    
#%%
    