# -*- coding: utf-8 -*-
"""
This code is to solve the 1-D neutron transport equation using T_N method.
@author: Longcong WANG
"""



import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

class TN:
    '''Class stores the T_N solver for 1-D slab reactor with isotropic scattering
    and isotropic source'''    
    def __init__(self,N):
        self.N = N #nth order approximation
    
    def geo(self,xi,xf,M,XS_t,XS_s,Q):
        '''definet the geometry and material properties'''
        self.xi = xi                    #left boundary of domain
        self.xf = xf                    #right boundary of domain
        self.x = np.linspace(xi,xf,M)   #generate mesh
        self.Q = Q                      #source term
        self.XS_t = XS_t                #total cross section
        self.XS_s = XS_s                #scattering cross section
        self.dx = (xf-xi)/(M-1)         #dx
        self.M = M                      #mesh size
        
    def H_nk(self,a,n):
        '''Coefficient function 1 for Marshak B.C.'''
        if a == 0:
            H_nk = (-1)**n/(2*n+1)
        else:   
            H_nk = ((-1)**(a+n)/(2*a+2*n+1)+
                (-1)**(a-n+1)/(2*a-2*n-1))
        return H_nk
        
               
    def I_nk(self,a,n):
        '''Coefficient function 2 for Marshak B.C.'''
        if a == 0 :
            I_nk = (-1)**(n+1)/(2*n+1)
        else:
            I_nk = ((-1)**(a+n+1)/(2*a+2*n+1)+
                (-1)**(a-n)/(2*a-2*n-1))      
        return I_nk
    
    def Q_2n(self,n):
        '''Solve the source term Q_n'''
        if n == 0:
            Q_factor = 2
        else:           
            Q_factor = 2/(1-(2*n)**2)
        return Q_factor
    
    def solve(self):
        '''Construct the matrixes and solve the linear algebra system'''
        N = self.N
        M = self.M
        dx = self.dx
        size_F = int((N-1)/2)        
        size = (size_F+1)*M
        A = np.zeros((size,size))
        Qv = np.zeros((size))
        dx2 = self.dx*self.dx        
        H_pre = 4*self.XS_t/dx/np.pi
        I_pre = -4*self.XS_t/dx/np.pi
        H_pre0 = 4*self.XS_t/dx/np.pi
        I_pre0 = -4*self.XS_t/dx/np.pi
        for n in range(0,size_F+1):
            Q_n = self.Q_2n(n)*self.Q
            if n == 0:
                Qv[:M] = 2*self.XS_t*Q_n
                for i in range(1,M-1):                    
                    A[n*M+i,n*M+i-1] += -1/dx2
                    A[n*M+i,n*M+i] += 2/dx2
                    A[n*M+i,n*M+i+1] += -1/dx2
                    for m in range(0,size_F-n+1):
                        A[n*M+i,m*M+i] += (2*self.XS_t*(self.XS_t-
                         self.XS_s)*(-1)**m)
                A[0,0] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
                A[M-1,M-1] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
                A[0,1] += -2/dx2
                A[M-1,M-2] += -2/dx2
                for a in range(0,size_F+1):
                    for m in range(0,size_F-a+1):
                        A[0,(m+a)*M] += H_pre0*self.H_nk(a,n)*(-1)**m
                        A[M-1,(m+a)*M+M-1] += I_pre0*self.I_nk(a,n)*(-1)**m
            else:              
                for i in range(1,M-1):
                    Qv[n*M+i] = 4*self.XS_t*Q_n
                    A[n*M+i,n*M+i] += 2/dx2
                    A[n*M+i,n*M+i-1] += -1/dx2
                    A[n*M+i,n*M+i+1] += -1/dx2
                    A[n*M+i,(n-1)*M+i] += 2/dx2
                    A[n*M+i,(n-1)*M+i+1] += -1/dx2
                    A[n*M+i,(n-1)*M+i-1] += -1/dx2
                    for m in range(0,size_F-n+1):
                        A[n*M+i,(n+m)*M+i] += (-1)**m*4*self.XS_t**2
                    for m in range(0,size_F+1):
                        A[n*M+i,m*M+i] += ((-1)**m*4*self.XS_t*self.XS_s
                         /(4*n*n-1))
                A[n*M,n*M] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
                A[n*M+M-1,n*M+M-1] += 2/dx2+2*XS_t**2-2*XS_t*XS_s
                A[n*M,n*M+1] += -2/dx2
                A[n*M+M-1,n*M+M-2] += -2/dx2
                for a in range(0,size_F+1):
                    for m in range(0,size_F-a+1):
                        A[n*M,(a+m)*M] += H_pre*self.H_nk(a,n)*(-1)**m
                        A[n*M+M-1,(a+m)*M+M-1] += I_pre*self.I_nk(a,n)*(-1)**m
        self.Qv = Qv
        self.A = A
        self.F = np.linalg.solve(A,Qv)
        self.phi0 = np.zeros((M))
        for m in range(0,size_F+1):
            '''solve the scalar flux phi_0'''
            self.phi0 += self.F[m*M:(m+1)*M]*(-1)**m

XS_t = 0.17
XS_s = 0.1
Q = 5
'''import results from P1 & P3'''
P1_phi0 = np.load('phi0_P1.npy')
P3_phi0 = np.load('phi0_P3.npy')
'''Conduct the order of accuracy convergence'''
T_N = []
T_N.append(TN(1))
T_N[0].geo(0,100,619,XS_t,XS_s,Q)
T_N[0].solve()
n = 1
error_set = []
error = 1 
while error >= 1e-3:
    T_N.append(TN(2*n+1))
    T_N[n].geo(0,100,619,XS_t,XS_s,Q)
    T_N[n].solve()
    error = np.max(np.abs(T_N[n].phi0-T_N[n-1].phi0)/T_N[n-1].phi0)
    error_set.append(error)
    print(error)
    n += 1
    
'''Visualization of results'''
n_plot = np.arange(3,2*n+1,2)
x_plot = np.linspace(0,100,619)
x_P = np.linspace(0,100,1001)

plt.figure()
plt.plot(n_plot,error_set)
plt.xlim([n_plot[0],n_plot[-1]])
plt.yscale("log")
plt.grid('True')
plt.xlabel(r'$N^{th}$ order approximation')
plt.ylabel('Reltive error')
plt.savefig('order_conv_2.pdf',dpi=1000)

plt.figure()
plt.plot(x_plot,T_N[n-1].phi0)
plt.xlim([x_plot[0],x_plot[-1]])
plt.grid('True')
plt.xlabel(r'x (m)')
plt.ylabel(r'$\phi_0$')
plt.savefig('phi0_final_2.pdf',dpi=1000)

plt.figure()
plt.plot(x_plot,T_N[0].phi0,'r',marker='o',markevery=20,markersize=3,lw=1,
         label=r'$T_1$')
plt.plot(x_plot,T_N[5].phi0,'b',marker='*',markevery=20,markersize=3,lw=1,
         label=r'$T_{11}$')
plt.plot(x_plot,T_N[-1].phi0,'g',marker='+',markevery=20,markersize=3,lw=1,
         label=r'$T_{23}$')
plt.plot(x_P,P1_phi0,'m',marker='<',markevery=20,markersize=3,lw=1,
         label=r'$P_1$')
plt.plot(x_P,P3_phi0,'c',marker='h',markevery=20,markersize=3,lw=1,
         label=r'$P_3$')
plt.xlim([0,100])
plt.grid('True')
plt.xlabel('x (cm)')
plt.ylabel(r'$\phi_0$')
plt.legend(loc=0)
plt.savefig('TP_comp_final_2.pdf',dpi=1000)