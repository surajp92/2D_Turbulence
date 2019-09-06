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

#%%
def export_resutls(y_test, y_pred, ilabel, nxf, nx, n, nn):
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
    
    folder = "data_"+ str(nxf) + "_" + str(nx) + "_V2"
    
    if not os.path.exists("spectral/"+folder+"/01_ML_DNN"):
        os.makedirs("spectral/"+folder+"/01_ML_DNN")
        os.makedirs("spectral/"+folder+"/01_ML_DNN/test_data_labels")
        os.makedirs("spectral/"+folder+"/01_ML_DNN/pred_data_labels")
        os.makedirs("spectral/"+folder+"/01_ML_CNN")
        os.makedirs("spectral/"+folder+"/01_ML_CNN/test_data_labels")
        os.makedirs("spectral/"+folder+"/01_ML_CNN/pred_data_labels")
        
    if nn == 1:
        folder = "data_"+ str(nxf) + "_" + str(nx) + "_V2/01_ML_DNN"
    elif nn == 2:
        folder = "data_"+ str(nxf) + "_" + str(nx) + "_V2/01_ML_CNN"
        
    if ilabel == 1:
        with open("spectral/"+folder+"/test_data_labels/y_test_sgs_"+str((n))+".csv", 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(y_test.shape))
            for data_slice in y_test:
                np.savetxt(outfile, data_slice, delimiter=",")
                outfile.write('# New slice\n')
        
        with open("spectral/"+folder+"/pred_data_labels/y_pred_sgs_"+str((n))+".csv", 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(y_pred.shape))
            for data_slice in y_pred:
                np.savetxt(outfile, data_slice, delimiter=",")
                outfile.write('# New slice\n')
    
    if ilabel == 2:
        filename = "spectral/"+folder+"/test_data_labels/y_test_nu_"+str((n))+".csv"
        np.savetxt(filename, y_test, delimiter=",")
        
        filename = "spectral/"+folder+"/pred_data_labels/y_pred_nu_"+str((n))+".csv"
        np.savetxt(filename, y_pred, delimiter=",")
        
#%%
def plot_dynamic_cs2(nxf, nx):
    folder = "data_"+ str(nxf) + "_" + str(nx) + "_V2"
            
    file_input = "spectral/"+folder+"/cs2/cs2.csv"
    data_input = np.genfromtxt(file_input, delimiter=',',skip_header=1)
    t = data_input[:,0]
    cs2 = data_input[:,2]
    
    fig, axs = plt.subplots(1, 1, figsize=(6,5))#, constrained_layout=True)
    axs.plot(t,cs2, color='k', lw = 4, linestyle='-', label=r'$C_s$', zorder=5)
    
#    axs.plot(t,cs2, color='navy', lw = 2, marker="o", linestyle='-', label=r'$C_s$', zorder=5)
#    axs[0].loglog(t,omega_54, color='orangered', marker="o", lw = 2, linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
#    axs[0].loglog(t,omega_74, color='purple', marker="o", lw = 2, linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
#    axs[0].loglog(t,omega_94, color='orange', marker="o", lw = 2, linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
    axs.grid(True)
    axs.set_xlim([t[0],t[int(t.shape[0]-1)]])
    axs.set_ylabel('$C_s$', fontsize = 14)
    axs.set_xlabel('$n$', fontsize = 14)
    
    fig.tight_layout() 
    fig.subplots_adjust(bottom=0.2)
    axs.legend()
    fig.savefig('cs.pdf')
    

#plot_dynamic_cs2(nxf=1024, nx=64)

#%%