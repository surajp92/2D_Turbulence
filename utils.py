#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:28:06 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
import os

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
        with open("spectral/"+folder+"/test_data_labels/y_test_sgs_"+str(int(n))+".csv", 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(y_test.shape))
            for data_slice in y_test:
                np.savetxt(outfile, data_slice, delimiter=",")
                outfile.write('# New slice\n')
        
        with open("spectral/"+folder+"/pred_data_labels/y_pred_sgs_"+str(int(n))+".csv", 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(y_pred.shape))
            for data_slice in y_pred:
                np.savetxt(outfile, data_slice, delimiter=",")
                outfile.write('# New slice\n')
    
    if ilabel == 2:
        filename = "spectral/"+folder+"/test_data_labels/y_test_nu_"+str(int(n))+".csv"
        np.savetxt(filename, y_test, delimiter=",")
        
        filename = "spectral/"+folder+"/pred_data_labels/y_pred_nu_"+str(int(n))+".csv"
        np.savetxt(filename, y_pred, delimiter=",")