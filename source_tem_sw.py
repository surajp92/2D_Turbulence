#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:05:26 2019

@author: Suraj Pawar
"""
import numpy as np
import pyfftw

#%%
def wave2phy(nx,ny,uf):
    
    '''
    Converts the field form frequency domain to the physical space.
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    uf : solution field in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    u : solution in physical space (along with periodic boundaries)
    '''
    
    u = np.empty((nx+1,ny+1))
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')

    u[0:nx,0:ny] = np.real(fft_object_inv(uf))
    # periodic BC
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    
    return u

#%%
def coarsen(nx,ny,nxc,nyc,uf):  
    
    '''
    coarsen the data along with the size of the data 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    uf : solution field on fine grid in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    u : caorsened solution in frequency domain (excluding periodic boundaries)
    '''
    
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc[0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]
    ufc[int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)]    
    ufc[0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):]
    ufc[int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    return ufc

       
#%%
def nonlineardealiased(nx,ny,kx,ky,k2,wf):    
    
    '''
    compute the Jacobian with 3/2 dealiasing 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kx,ky : wavenumber in x and y direction
    k2 : absolute wave number over 2D domain
    wf : vorticity field in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    jf : jacobian in frequency domain (excluding periodic boundaries)
         (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    '''
    
    j1f = 1.0j*kx*wf/k2
    j2f = 1.0j*ky*wf
    j3f = 1.0j*ky*wf/k2
    j4f = 1.0j*kx*wf
    
    nxe = int(nx*3/2)
    nye = int(ny*3/2)
    
    j1f_padded = np.zeros((nxe,nye),dtype='complex128')
    j2f_padded = np.zeros((nxe,nye),dtype='complex128')
    j3f_padded = np.zeros((nxe,nye),dtype='complex128')
    j4f_padded = np.zeros((nxe,nye),dtype='complex128')
    
    j1f_padded[0:int(nx/2),0:int(ny/2)] = j1f[0:int(nx/2),0:int(ny/2)]
    j1f_padded[int(nxe-nx/2):,0:int(ny/2)] = j1f[int(nx/2):,0:int(ny/2)]    
    j1f_padded[0:int(nx/2),int(nye-ny/2):] = j1f[0:int(nx/2),int(ny/2):]    
    j1f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j1f[int(nx/2):,int(ny/2):] 
    
    j2f_padded[0:int(nx/2),0:int(ny/2)] = j2f[0:int(nx/2),0:int(ny/2)]
    j2f_padded[int(nxe-nx/2):,0:int(ny/2)] = j2f[int(nx/2):,0:int(ny/2)]    
    j2f_padded[0:int(nx/2),int(nye-ny/2):] = j2f[0:int(nx/2),int(ny/2):]    
    j2f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j2f[int(nx/2):,int(ny/2):] 
    
    j3f_padded[0:int(nx/2),0:int(ny/2)] = j3f[0:int(nx/2),0:int(ny/2)]
    j3f_padded[int(nxe-nx/2):,0:int(ny/2)] = j3f[int(nx/2):,0:int(ny/2)]    
    j3f_padded[0:int(nx/2),int(nye-ny/2):] = j3f[0:int(nx/2),int(ny/2):]    
    j3f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j3f[int(nx/2):,int(ny/2):] 
    
    j4f_padded[0:int(nx/2),0:int(ny/2)] = j4f[0:int(nx/2),0:int(ny/2)]
    j4f_padded[int(nxe-nx/2):,0:int(ny/2)] = j4f[int(nx/2):,0:int(ny/2)]    
    j4f_padded[0:int(nx/2),int(nye-ny/2):] = j4f[0:int(nx/2),int(ny/2):]    
    j4f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j4f[int(nx/2):,int(ny/2):] 
    
    j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
    j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
    j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
    j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
    
    
    a = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    fft_object_inv1 = pyfftw.FFTW(a1, b1,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv2 = pyfftw.FFTW(a2, b2,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv3 = pyfftw.FFTW(a3, b3,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv4 = pyfftw.FFTW(a4, b4,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    j1 = np.real(fft_object_inv1(j1f_padded))
    j2 = np.real(fft_object_inv2(j2f_padded))
    j3 = np.real(fft_object_inv3(j3f_padded))
    j4 = np.real(fft_object_inv4(j4f_padded))
    
    jacp = j1*j2 - j3*j4
    
    jacpf = fft_object(jacp)
    
    jf = np.zeros((nx,ny),dtype='complex128')
    
    jf[0:int(nx/2),0:int(ny/2)] = jacpf[0:int(nx/2),0:int(ny/2)]
    jf[int(nx/2):,0:int(ny/2)] = jacpf[int(nxe-nx/2):,0:int(ny/2)]    
    jf[0:int(nx/2),int(ny/2):] = jacpf[0:int(nx/2),int(nye-ny/2):]    
    jf[int(nx/2):,int(ny/2):] =  jacpf[int(nxe-nx/2):,int(nye-ny/2):]
    
    jf = jf*(nx*ny)/(nxe*nye)
    
    return jf

#%%
def sourceterm(nx,ny,nxc,nyc,wf,n):
    
    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    k2 = kx*kx + ky*ky
    k2[0,0] = 1.0e-12

    kxc = np.fft.fftfreq(nxc,1/nxc)
    kyc = np.fft.fftfreq(nyc,1/nyc)
    kxc = kxc.reshape(nxc,1)
    kyc = kyc.reshape(1,nyc)
    
    k2c = kxc*kxc + kyc*kyc
    k2c[0,0] = 1.0e-12
     
    jf = nonlineardealiased(nx,ny,kx,ky,k2,wf)

    jc = np.zeros((nxc+1,nyc+1)) # coarsened(jacobian field)
    jfc = coarsen(nx,ny,nxc,nyc,jf) # coarsened(jacobian field) in frequency domain
    jc = wave2phy(nxc,nyc,jfc) # coarsened(jacobian field) physical space
       
    wfc = coarsen(nx,ny,nxc,nyc,wf)       
    jcoarsef = nonlineardealiased(nxc,nyc,kxc,kyc,k2c,wfc) # jacobian(coarsened solution field) in frequency domain
    jcoarse = wave2phy(nxc,nyc,jcoarsef) # jacobian(coarsened solution field) physical space
    
    sgs = jcoarse - jc
    
    folder = "data_"+ str(nx) + "_V2" 
    filename = "spectral/"+folder+"/00_sgs/sgs_"+str(int(n))+".csv"
    np.savetxt(filename, sgs, delimiter=",")
    
#%%
l1 = []
with open('input_aprior.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ich = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])
alpha = np.float64(l1[10][0])

freq = int(nt/ns)

if (ich != 19):
    print("Check input.txt file")

#%%
nx = nd
ny = nd

nxc = ndc
nyc = ndc

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

#%%
for n in range(1,ns+1):
    folder = "data_"+str(nx)
    file_input = "spectral/"+folder+"/04_vorticity/w_"+str(n)+".csv"
    w = np.genfromtxt(file_input, delimiter=',')    
    
    data = np.vectorize(complex)(w[0:nx,0:ny],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    wf = fft_object(data) # fourier space forward
    
    sourceterm(nx,ny,nxc,nyc,wf,n)