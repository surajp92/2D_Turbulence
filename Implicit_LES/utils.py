#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:45:02 2020

@author: suraj
"""
import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os


font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx X ny]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx
    uy : du/dy
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
def dyn_smag(nx,ny,kappa,sc,wc):
    
    nxc = int(nx/kappa) 
    nyc = int(ny/kappa)
    
    scc = les_filter(nx,ny,nxc,nyc,sc[2:nx+3,2:ny+3])
    wcc = les_filter(nx,ny,nxc,nyc,wc[2:nx+3,2:ny+3])
    
    scx,scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
    wcx,wcy = grad_spectral(nx,ny,wc[2:nx+3,2:ny+3])
    
    wcxx,wcxy = grad_spectral(nx,ny,wcx)
    wcyx,wcyy = grad_spectral(nx,ny,wcy)
    
    scxx,scxy = grad_spectral(nx,ny,scx)
    scyx,scyy = grad_spectral(nx,ny,scy)
    
    dac = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2) # |\bar(s)|
    dacc = les_filter(nx,ny,nxc,nyc,dac)        # |\tilde{\bar{s}}| = \tilde{|\bar(s)|}
    
    sccx,sccy = grad_spectral(nx,ny,scc)
    wccx,wccy = grad_spectral(nx,ny,wcc)
    
    wccxx,wccxy = grad_spectral(nx,ny,wccx)
    wccyx,wccyy = grad_spectral(nx,ny,wccy)
    
    scy_wcx = scy*wcx
    scx_wcy = scx*wcy
    
    scy_wcx_c = les_filter(nx,ny,nxc,nyc,scy_wcx)
    scx_wcy_c = les_filter(nx,ny,nxc,nyc,scx_wcy)
    
    h = (sccy*wccx - sccx*wccy) - (scy_wcx_c - scx_wcy_c)
    
    t = dac*(wcxx + wcyy)
    tc = les_filter(nx,ny,nxc,nyc,t)
    
    m = kappa**2*dacc*(wccxx + wccyy) - tc
    
    hm = h*m
    mm = m*m
    
    CS2 = (np.sum(0.5*(hm + abs(hm)))/np.sum(mm))
    
    ev = CS2*dac
    
    return ev

    
#%%
#def dyn_smag(nx,ny,nxc,nyc,dx,dy,sc,wc):
#        
#    scx, scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
#    scxx, scxy =  grad_spectral(nx,ny,scx)
#    scyx, scyy =  grad_spectral(nx,ny,scy)
#    
#    CS2 = compute_cs(nx,ny,sc,wc)
#    
#    ev = CS2*np.sqrt(4.0*scxy**2 + (scxx-scyy)**2)
    
    
#    dsdxy = (1.0/(4.0*dx*dy))*(sc[1:nx+2,1:ny+2] + sc[3:nx+4,3:ny+4] \
#                              -sc[3:nx+4,1:ny+2] - sc[1:nx+2,3:ny+4])
#    
#    dsdxx = (1.0/(dx*dx))*(sc[3:nx+4,2:ny+3] - 2.0*sc[2:nx+3,2:ny+3] \
#                                         +sc[1:nx+2,2:ny+3])
#    
#    dsdyy = (1.0/(dy*dy))*(sc[2:nx+3,3:ny+4] - 2.0*sc[2:nx+3,2:ny+3] \
#                                         +sc[2:nx+3,1:ny+2])
    
#    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
#    return ev
