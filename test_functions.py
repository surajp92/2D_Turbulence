#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:35:43 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
def coarsen(nx,ny,nxc,nyc,u):
    uf = np.fft.fft2(u[0:nx,0:ny])#, norm = "ortho")
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc [0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]     
    ufc [int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)] 
    ufc [0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):] 
    ufc [int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    utc = np.real(np.fft.ifft2(ufc))
    
    uc = np.zeros((nxc+1,nyc+1))
    uc[0:nxc,0:nyc] = utc
    uc[:,nyc] = uc[:,0]
    uc[nxc,:] = uc[0,:]
    uc[nxc,nyc] = uc[0,0]
    
    return uc
    
#%%
m = 400
nx, ny = 1024, 1024
nxc1, nyc1 = 64, 64
nxc2, nyc2 = 128, 128
file_input = "spectral/data_1024/05_streamfunction/s_"+str(m)+".csv"
s = np.genfromtxt(file_input, delimiter=',')
sx,sy = grad_spectral(nx,ny,s)
u = sy
v = -sx

uu = u*u
uv = u*v
vv = v*v

#%%
uuc = coarsen(nx,ny,nxc1,nyc1,uu)
uvc = coarsen(nx,ny,nxc1,nyc1,uv)
vvc = coarsen(nx,ny,nxc1,nyc1,vv)
uc = coarsen(nx,ny,nxc1,nyc1,u)
vc = coarsen(nx,ny,nxc1,nyc1,v)

t11c = uuc - uc*uc
t12c = uvc - uc*vc
t22c = vvc - vc*vc

#%%
uuc2 = coarsen(nx,ny,nxc2,nyc2,uu)
uvc2 = coarsen(nx,ny,nxc2,nyc2,uv)
vvc2 = coarsen(nx,ny,nxc2,nyc2,vv)
uc2 = coarsen(nx,ny,nxc2,nyc2,u)
vc2 = coarsen(nx,ny,nxc2,nyc2,v)

t11c2 = uuc2 - uc2*uc2
t12c2 = uvc2 - uc2*vc2
t22c2 = vvc2 - vc2*vc2

#%%
x = np.linspace(0,2*np.pi,nx+1)
y = np.linspace(0,2*np.pi,ny+1)

x = x.reshape(1,nx+1)
y = y.reshape(ny+1,1)

f = np.sin(x) + np.cos(y)

f1 = coarsen(nx,ny,nxc1,nyc1,f)
f2 = coarsen(nx,ny,nxc2,nyc2,f)

#%% PDF plotting
num_bins = 64

fig, axs = plt.subplots(2,3,figsize=(12,7))
axs[0,0].set_yscale('log')
axs[0,1].set_yscale('log')
axs[0,2].set_yscale('log')

# the histogram of the data
ntrue, binst, patchest = axs[0,0].hist(uc.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[0,0].hist(uc2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[0,1].hist(vc.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[0,1].hist(vc2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[1,0].hist(uuc.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[1,0].hist(uuc2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[1,1].hist(uvc.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[1,1].hist(uvc2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[1,2].hist(vvc.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[1,2].hist(vvc2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

axs[0,0].set_title("u")
axs[0,1].set_title("v")
axs[1,0].set_title("uu")
axs[1,1].set_title("uv")
axs[1,2].set_title("vv")
axs[0,0].legend()
fig.tight_layout()
plt.show()

#%%
num_bins = 64

fig, axs = plt.subplots(1,3,figsize=(12,4))
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

# the histogram of the data
ntrue, binst, patchest = axs[0].hist(t11c.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[0].hist(t11c2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[1].hist(t12c.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[1].hist(t12c2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

ntrue, binst, patchest = axs[2].hist(t22c.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,density=True,label="64")
npred, binsp, patchesp = axs[2].hist(t22c2.flatten(), num_bins, histtype='step', alpha=1,color='b',zorder=10,
                                 linewidth=2.0,density=True,label="256")

axs[0].set_title("t11")
axs[1].set_title("t12")
axs[2].set_title("t22")
axs[0].legend()

fig.tight_layout()
plt.show()
#%% DNN data generation
n_snapshots  = 50
n_snapshots_test = 10
n_snapshots_train = n_snapshots - n_snapshots_test
for m in range(1,n_snapshots_train):
    file_input = "spectral/data_1024/uc/uc_"+str(m)+".csv"
    uc = np.genfromtxt(file_input, delimiter=',')
    file_input = "spectral/data_1024/vc/vc_"+str(m)+".csv"
    vc = np.genfromtxt(file_input, delimiter=',')
    file_input = "spectral/data_1024/nu/nu_"+str(m)+".csv"
file_input = "spectral/data_1024/true_shear_stress/t_"+str(m)+".csv"
t = np.genfromtxt(file_input, delimiter=',')
t = t.reshape((3,65,65))
t11c = t[0,:,:]
t12c = t[1,:,:]
t22c = t[2,:,:]    
nu = np.genfromtxt(file_input, delimiter=',')
    
nx,ny = uc.shape
nt = int((nx-2)*(ny-2))

x_t = np.zeros((nt,18))
y_t = np.zeros((nt,1))

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
        
        y_t[n,0] = nu[i,j]
        n = n+1

if m == 1:
    x_train = x_t
    y_train = y_t
else:
    x_train = np.vstack((x_train,x_t))
    y_train = np.vstack((y_train,y_t))

#%% field plotting
nx,ny = 64,64
m = 400
file_input = "spectral/data_1024_64/nu_true/nut_"+str(m)+".csv"
nu = np.genfromtxt(file_input, delimiter=',')
nu = nu.reshape((3,nx+1,ny+1))
nu11 = nu[0,:,:]
nu12 = nu[1,:,:]
nu22 = nu[2,:,:]

file_input = "spectral/data_1024_64/true_shear_stress/t_"+str(m)+".csv"
t = np.genfromtxt(file_input, delimiter=',')
t = t.reshape((3,nx+1,ny+1))
t11 = t[0,:,:]
t12 = t[1,:,:]
t22 = t[2,:,:]

#%%
file_input = "spectral/data_1024/04_vorticity/w_"+str(m)+".csv"
w = np.genfromtxt(file_input, delimiter=',')

file_input = "spectral/data_1024_64/00_wc/wc_"+str(m)+".csv"
wc = np.genfromtxt(file_input, delimiter=',')

fig, axs = plt.subplots(1,2,figsize=(10,4))

cs = axs[0].contourf(w.T,120, cmap = 'jet', interpolation='bilinear')
fig.colorbar(cs, ax = axs[0])
axs[0].set_title(r"$\omega$")

cs = axs[1].contourf(wc.T,120, cmap = 'jet', interpolation='bilinear')
fig.colorbar(cs, ax = axs[1])
axs[1].set_title(r"$\omega_c$")

plt.show()
fig.savefig('vorticity.pdf')

#%%
fig, axs = plt.subplots(2,3,sharey=True,figsize=(12,6))

v = np.linspace(-1, 1., 8, endpoint=True)

cs = axs[0,0].contourf(nu11.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[0,0])
axs[0,0].set_title(r"$\nu_{11}$")

cs = axs[0,1].contourf(nu12.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[0,1])
axs[0,1].set_title(r"$\nu_{12}$")

cs = axs[0,2].contourf(nu22.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[0,2])
axs[0,2].set_title(r"$\nu_{22}$")

v11 = np.linspace(-0.2, 0.2, 8, endpoint=True)
v12 = np.linspace(-0.2, 0.2, 8, endpoint=True)
v22 = np.linspace(-0.2, 0.2, 8, endpoint=True)

cs = axs[1,0].contourf(t11.T,levels=v11,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[1,0])
axs[1,0].set_title(r"$\tau_{11}$")

cs = axs[1,1].contourf(t12.T,levels=v12,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[1,1])
axs[1,1].set_title(r"$\tau_{12}$")

cs = axs[1,2].contourf(t22.T,levels=v22,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[1,2])
axs[1,2].set_title(r"$\tau_{22}$")

plt.show()
fig.savefig('nu_stresses.pdf')

#%%
file_input = "spectral/data_1024_64/ucx/ucx_"+str(m)+".csv"
ucx = np.genfromtxt(file_input, delimiter=',')

file_input = "spectral/data_1024_64/ucy/ucy_"+str(m)+".csv"
ucy = np.genfromtxt(file_input, delimiter=',')

file_input = "spectral/data_1024_64/vcx/vcx_"+str(m)+".csv"
vcx = np.genfromtxt(file_input, delimiter=',')

file_input = "spectral/data_1024_64/vcy/vcy_"+str(m)+".csv"
vcy = np.genfromtxt(file_input, delimiter=',')

fig, axs = plt.subplots(2,2,sharey=True,figsize=(10,8))

v = np.linspace(-6, 6., 8, endpoint=True)
cs = axs[0,0].contourf(ucx.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[0,0])
axs[0,0].set_title(r"$\frac{\partial u_c}{\partial x}$")

v = np.linspace(-10, 10., 8, endpoint=True)
cs = axs[0,1].contourf(ucy.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[0,1])
axs[0,1].set_title(r"$\frac{\partial u_c}{\partial y}$")

v = np.linspace(-14, 14., 8, endpoint=True)
cs = axs[1,0].contourf(vcx.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[1,0])
axs[1,0].set_title(r"$\frac{\partial v_c}{\partial x}$")

v = np.linspace(-7, 7., 8, endpoint=True)
cs = axs[1,1].contourf(vcy.T,levels=v,cmap=cm.jet,extend="both")
fig.colorbar(cs,ticks = v,ax = axs[1,1])
axs[1,1].set_title(r"$\frac{\partial v_c}{\partial y}$")

plt.show()
fig.savefig('velocity_gradients.pdf')

#%%
freq = 1
n_snapshots = 400
nxf, nyf = 1024, 1024
nx, ny = 64, 64

folder = "data_"+ str(nxf) + "_" + str(nx) + "_V2" 
m = n_snapshots*freq

file_input = "spectral/"+folder+"/L/L_"+str(m)+".csv"
LM = np.genfromtxt(file_input, delimiter=',')

file_input = "spectral/"+folder+"/M/M_"+str(m)+".csv"
MM = np.genfromtxt(file_input, delimiter=',')

x = MM.flatten()
y = LM.flatten()

#%%
fig, axs = plt.subplots(1,2,figsize=(11,4))

axs[0].loglog(x,y,lw=0.0,marker="o")
axs[0].set_xlabel(r"$<M_{ij}M_{ij}>$")
axs[0].set_ylabel(r"$<L_{ij}M_{ij}>$")

axs[1].scatter(MM,abs(LM))
axs[1].set_yscale('log')
axs[1].set_xscale('log')

axs[1].set_xlim([10e-6,10e1])
axs[1].set_ylim([10e-6,10e0])

#axs.legend(loc='upper left')

plt.show()









