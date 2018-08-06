# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:39:40 2018

@author: Longcong WANG
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

    
def solve(x0,x1,M):
    
    dx = (x1-x0)/(M-1)
    dx2 = dx*dx
    Q_0 = 10
    XS_t = 0.17
    XS_s = 0.1
    Q = 2*XS_t*Q_0*np.ones((M))
    A = np.zeros((M,M))
    
    for i in range(1,M-1):
        A[i,i] = 2/dx2+2*XS_t**2-2*XS_t*XS_s
        A[i,i-1] = -1/dx2
        A[i,i+1] = -1/dx2
    
    A[0,0] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
    A[0,1] += -2/dx2
    A[0,0] += 4/dx2*dx*XS_t/np.pi
    A[M-1,M-1] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
    A[M-1,M-2] += -2/dx2
    A[M-1,M-1] += 4/dx2*dx*XS_t/np.pi
    F_0 = np.linalg.solve(A,Q)
    return F_0

error = 1
error_set = []
M=10
M_set = []
F_prev = solve(0,100,M)
while error >= 1e-6:
    M += 1
    F_0 = solve(0,100,M)
    error = np.abs((np.average(F_prev)-np.average(F_0))/np.average(F_prev))
    error_set.append(error)
    F_prev = F_0
    M_set.append(M)

P1_phi0 = np.load('phi0_P1.npy')
P3_phi0 = np.load('phi0_P3.npy')   
x = np.linspace(0,100,M)
x_P = np.linspace(0,100,1001)
'''Visualization of results'''
plt.figure()
plt.plot(x,F_0,'r',marker='o',markevery=20,markersize=3,lw=1,
         label=r'$T_1$')
plt.plot(x_P,P1_phi0,'b',marker='*',markevery=20,markersize=3,lw=1,
         label=r'$P_1$')
plt.plot(x_P,P3_phi0,'g',marker='+',markevery=20,markersize=3,lw=1,
         label=r'$P_3$')
plt.xlim([0,100])
plt.grid('True')
plt.xlabel('x (cm)')
plt.ylabel(r'$\phi_0$')
plt.legend(loc=0)
plt.savefig('TP_comp_2.pdf',dpi=1000)

plt.figure()
plt.plot(x,F_0)
plt.xlim([0,100])
plt.grid('True')
plt.xlabel('x (cm)')
plt.ylabel(r'$\phi_0$')
plt.savefig('T1_final_2.pdf',dpi=1000)

plt.figure()
plt.plot(M_set,error_set)
plt.xlim([M_set[0],M_set[-1]])
plt.yscale("log")
plt.grid('True')
plt.xlabel('Mesh size')
plt.ylabel('Reltive error')
plt.savefig('conv_mesh_2.pdf',dpi=1000)
plt.savefig('conv_mesh_2.png',dpi=1000)