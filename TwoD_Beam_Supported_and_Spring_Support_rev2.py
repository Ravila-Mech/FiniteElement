# Prepare environment and import libraries

import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Functions definition
def Mesh1D(L1, Nx):
    # Generates nodes positions and connectivity table for 1D mesh of length L1 
    # and number of elements Nx
    # Linear elements only
    # Nodes array contains nodal positions (one node per row)
    # Connectivity array contains the element nodes number (one element per each row)
    
    Nodes = np.zeros(shape=(Nx+1, 1))
    Connectivity = np.zeros(shape=(Nx, 2), dtype=int)
    
    DeltaX = L1/Nx
    for i in range(0, Nx+1):
        Nodes[i,0] = i*DeltaX
        
    for i in range(0, Nx):
        Connectivity[i,0] = i
        Connectivity[i,1] = i+1
        
    return Nodes, Connectivity



def Quadrature1D(order):
    
    if order == 1:
        WQ = [2]
        XIq = [0]
    elif order == 2:
        WQ = [1, 1]
        XIq = [-1/np.sqrt(3), 1/np.sqrt(3)]
    elif order == 3:
        WQ = [8/9, 5/9, 5/9]
        XIq = [0,np.sqrt(15)/5, -np.sqrt(15)/5]       
    elif order == 4:
        WQ = [(18+np.sqrt(30))/36,(18+np.sqrt(30))/36,(18-np.sqrt(30))/36,(18-np.sqrt(30))/36]
        XIq = [np.sqrt(3/7-(2/7*np.sqrt(6/5))),-np.sqrt(3/7-(2/7*np.sqrt(6/5))),np.sqrt(3/7+(2/7*np.sqrt(6/5))),-np.sqrt(3/7+(2/7*np.sqrt(6/5)))]

    else:
        print("Error: quadrature rule not implemented")
    
    return WQ, XIq



def Shape1D(xiq, type):
    
    if type == 1:
        
        N = [0.5*(1.0 - xiq), 0.5*(1.0 + xiq)]; 
        
        DN = [-0.5, 0.5];
        
        DDN = 0

    elif type == 2:
        
        N = [0.25*((1-xiq)**2)*(2+xiq), 0.25*((1-xiq)**2)*(1+xiq), 0.25*((1+xiq)**2)*(2-xiq), 0.25*((1+xiq)**2)*(xiq-1)];
               
        DN = [0.75*((xiq**2)-1), 0.75*(xiq**2)-0.5*xiq-0.25, 0.75*(1-xiq**2), 0.75*(xiq**2)+0.5*xiq-0.25];
        
        DDN = [1.5*xiq, 1.5*xiq-0.5, -1.5*xiq, 1.5*xiq+0.5];
               
    else:
        print("Error: type of shape function not implemented")
    
    return N, DN , DDN
    


def LinElement1D(Nodes_el, E, I, A, rho, qy, px):

    
    WQ1, XIq1 = Quadrature1D(1)
    
    N, DN , DDN = Shape1D(XIq1[0],1)
    
    # Compute shape functions derivatives in the X domain
    DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1];  
    DNDX  = [DN[0]/DXDxi, DN[1]/DXDxi]
    
    # Compute k and m matrix
    K_el = np.zeros(shape=(6, 6))
    M_el = np.zeros(shape=(6, 6))
    F_el = np.zeros(shape=(6, 1))  
    
    K_el[0,0] = E*A * DNDX[0]*DNDX[0] * DXDxi * WQ1[0]
    K_el[0,3] = E*A * DNDX[0]*DNDX[1] * DXDxi * WQ1[0]
    K_el[3,0] = E*A * DNDX[1]*DNDX[0] * DXDxi * WQ1[0]
    K_el[3,3] = E*A * DNDX[1]*DNDX[1] * DXDxi * WQ1[0]
    
    WQ2, XIq2 = Quadrature1D(2)
        
    for q in range(0, np.size(WQ2)):
        
        N, DN ,DDN= Shape1D(XIq2[q],2)
        
        DXDxi = (Nodes_el[1] - Nodes_el[0])/2
        DDNDX = [DDN[0]/DXDxi/DXDxi, DDN[1]/DXDxi, DDN[2]/DXDxi/DXDxi, DDN[3]/DXDxi]
    
        K_el[1,1] += E*I * DDNDX[0]*DDNDX[0] * DXDxi * WQ2[q]
        K_el[1,2] += E*I * DDNDX[0]*DDNDX[1] * DXDxi * WQ2[q]       
        K_el[1,4] += E*I * DDNDX[0]*DDNDX[2] * DXDxi * WQ2[q]
        K_el[1,5] += E*I * DDNDX[0]*DDNDX[3] * DXDxi * WQ2[q]    
        K_el[2,1] += E*I * DDNDX[1]*DDNDX[0] * DXDxi * WQ2[q]
        K_el[2,2] += E*I * DDNDX[1]*DDNDX[1] * DXDxi * WQ2[q]       
        K_el[2,4] += E*I * DDNDX[1]*DDNDX[2] * DXDxi * WQ2[q]
        K_el[2,5] += E*I * DDNDX[1]*DDNDX[3] * DXDxi * WQ2[q]
        K_el[4,1] += E*I * DDNDX[2]*DDNDX[0] * DXDxi * WQ2[q]
        K_el[4,2] += E*I * DDNDX[2]*DDNDX[1] * DXDxi * WQ2[q]       
        K_el[4,4] += E*I * DDNDX[2]*DDNDX[2] * DXDxi * WQ2[q]
        K_el[4,5] += E*I * DDNDX[2]*DDNDX[3] * DXDxi * WQ2[q]    
        K_el[5,1] += E*I * DDNDX[3]*DDNDX[0] * DXDxi * WQ2[q]
        K_el[5,2] += E*I * DDNDX[3]*DDNDX[1] * DXDxi * WQ2[q]       
        K_el[5,4] += E*I * DDNDX[3]*DDNDX[2] * DXDxi * WQ2[q]
        K_el[5,5] += E*I * DDNDX[3]*DDNDX[3] * DXDxi * WQ2[q] 
    
    WQ3, XIq3 = Quadrature1D(2)
        
    for p in range(0, np.size(WQ3)):
        
        N, DN ,DDN= Shape1D(XIq3[p],1)
        
        DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1]; 
  
        M_el[0,0] += rho*A * N[0]*N[0] * DXDxi * WQ3[p]
        M_el[0,3] += rho*A * N[0]*N[1] * DXDxi * WQ3[p]
        M_el[3,0] += rho*A * N[1]*N[0] * DXDxi * WQ3[p]
        M_el[3,3] += rho*A * N[1]*N[1] * DXDxi * WQ3[p]      

    WQ4, XIq4 = Quadrature1D(4)  

    for r in range(0, np.size(WQ4)):

        N, DN ,DDN= Shape1D(XIq4[r],2)
        DXDxi = (Nodes_el[1] - Nodes_el[0])/2  
        N[1] = N[1]*DXDxi
        N[3] = N[3]*DXDxi
        
        M_el[1,1] += rho*A * N[0]*N[0] * DXDxi * WQ4[r]
        M_el[1,2] += rho*A * N[0]*N[1] * DXDxi * WQ4[r]       
        M_el[1,4] += rho*A * N[0]*N[2] * DXDxi * WQ4[r]
        M_el[1,5] += rho*A * N[0]*N[3] * DXDxi * WQ4[r] 
        M_el[2,1] += rho*A * N[1]*N[0] * DXDxi * WQ4[r]
        M_el[2,2] += rho*A * N[1]*N[1] * DXDxi * WQ4[r]       
        M_el[2,4] += rho*A * N[1]*N[2] * DXDxi * WQ4[r]
        M_el[2,5] += rho*A * N[1]*N[3] * DXDxi * WQ4[r] 
        M_el[4,1] += rho*A * N[2]*N[0] * DXDxi * WQ4[r]
        M_el[4,2] += rho*A * N[2]*N[1] * DXDxi * WQ4[r]       
        M_el[4,4] += rho*A * N[2]*N[2] * DXDxi * WQ4[r]
        M_el[4,5] += rho*A * N[2]*N[3] * DXDxi * WQ4[r] 
        M_el[5,1] += rho*A * N[3]*N[0] * DXDxi * WQ4[r]
        M_el[5,2] += rho*A * N[3]*N[1] * DXDxi * WQ4[r]       
        M_el[5,4] += rho*A * N[3]*N[2] * DXDxi * WQ4[r]
        M_el[5,5] += rho*A * N[3]*N[3] * DXDxi * WQ4[r]    
       
    WQ5, XIq5 = Quadrature1D(1)
        
    for s in range(0, np.size(WQ5)): 
        
        N, DN ,DDN= Shape1D(XIq5[s],1)

        DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1]; 

        F_el[0] += px * N[0] * DXDxi * WQ5[s]
        F_el[3] += px * N[1] * DXDxi * WQ5[s]
 
    WQ6, XIq6 = Quadrature1D(3)
        
    for o in range(0, np.size(WQ6)): 
        
        N, DN ,DDN= Shape1D(XIq6[o],2)

        DXDxi = (Nodes_el[1] - Nodes_el[0])/2  
        N[1] = N[1]*DXDxi
        N[3] = N[3]*DXDxi

        F_el[1] += qy * N[0] * DXDxi * WQ6[o]
        F_el[2] += qy * N[1] * DXDxi * WQ6[o] 
        F_el[4] += qy * N[2] * DXDxi * WQ6[o]
        F_el[5] += qy * N[3] * DXDxi * WQ6[o] 
        
    return  M_el, K_el, F_el
    
def FullMatrix(M_el, K_el, F_el,Nx,pl)    :
    M = np.zeros(shape=(3*Nx+3, 3*Nx+3))
    K = np.zeros(shape=(3*Nx+3, 3*Nx+3))
    F = np.zeros(shape=(3*Nx+3, 1))
    
    for p in range(1, Nx+1):
        M[3*p-3:3*p+3,3*p-3:3*p+3] = M[3*p-3:3*p+3,3*p-3:3*p+3] + M_el
        K[3*p-3:3*p+3,3*p-3:3*p+3] = K[3*p-3:3*p+3,3*p-3:3*p+3] + K_el
        F[3*p-3:3*p+3] = F[3*p-3:3*p+3] + F_el
   
    l = (3*Nx+4)/2
    ll = int(l)
    F[ll] = F[ll] + pl
        
    return M, K, F

def Constraint(M, K, F, k_spring):
    
    M_s = M
    M_s = np.delete(M_s,3*Nx+2,0)
    M_s = np.delete(M_s,3*Nx+1,0)
    M_s = np.delete(M_s,3*Nx,0)
    M_s = np.delete(M_s,3*Nx+2,1)
    M_s = np.delete(M_s,3*Nx+1,1)
    M_s = np.delete(M_s,3*Nx,1)    
    M_s = np.delete(M_s,2,0)
    M_s = np.delete(M_s,1,0)
    M_s = np.delete(M_s,0,0)
    M_s = np.delete(M_s,2,1)
    M_s = np.delete(M_s,1,1)
    M_s = np.delete(M_s,0,1)       
    K_s = K
    K_s = np.delete(K_s,3*Nx+2,0)
    K_s = np.delete(K_s,3*Nx+1,0)
    K_s = np.delete(K_s,3*Nx,0)
    K_s = np.delete(K_s,3*Nx+2,1)
    K_s = np.delete(K_s,3*Nx+1,1)
    K_s = np.delete(K_s,3*Nx,1)    
    K_s = np.delete(K_s,2,0)
    K_s = np.delete(K_s,1,0)
    K_s = np.delete(K_s,0,0)
    K_s = np.delete(K_s,2,1)
    K_s = np.delete(K_s,1,1)
    K_s = np.delete(K_s,0,1) 
    F_s = F
    F_s = np.delete(F_s,3*Nx+2,0)
    F_s = np.delete(F_s,3*Nx+1,0)
    F_s = np.delete(F_s,3*Nx,0)      
    F_s = np.delete(F_s,2,0)
    F_s = np.delete(F_s,1,0)
    F_s = np.delete(F_s,0,0)
    
    M_sp = M
    M_sp = np.delete(M_sp,3*Nx,0)
    M_sp = np.delete(M_sp,3*Nx,1)  
    M_sp = np.delete(M_sp,0,0)
    M_sp = np.delete(M_sp,0,1)     
    K_sp = K
    K_sp = np.delete(K_sp,3*Nx,0)
    K_sp = np.delete(K_sp,3*Nx,1) 
    K_sp = np.delete(K_sp,0,0)
    K_sp = np.delete(K_sp,0,1)
    K_sp[0,0] = K_sp[0,0] + k_spring
    K_sp[3*Nx-1,3*Nx-1] = K_sp[3*Nx-1,3*Nx-1] + k_spring
    F_sp = F
    F_sp = np.delete(F_sp,3*Nx,0)   
    F_sp = np.delete(F_sp,0,0)
    
    u_s = np.linalg.solve(K_s, F_s)
    u_sp = np.linalg.solve(K_sp, F_sp)
    return M_s, K_s, F_s, M_sp, K_sp, F_sp, u_s, u_sp

def Displacements(u_s, u_sp):
    u_xs = u_s
    u_ys = u_s
    
    for j in range(1,Nx):
        u_xs = np.delete(u_xs,3*(Nx-j)-1,0)
        u_xs = np.delete(u_xs,3*(Nx-j)-2,0)
        u_ys = np.delete(u_ys,3*(Nx-j)-1,0)
        u_ys = np.delete(u_ys,3*(Nx-1-j),0)
    
    U_xs = np.zeros(shape=(Nx+1, 1))
    U_ys = np.zeros(shape=(Nx+1, 1))
    U_xs[1:Nx] = U_xs[1:Nx] + u_xs
    U_ys[1:Nx] = U_ys[1:Nx] + u_ys
    
    
    u_xsp = u_sp
    u_xsp = np.delete(u_xsp,3*Nx,0)
    u_xsp = np.delete(u_xsp,3*Nx-1,0)
    u_xsp =np.delete(u_xsp,1,0)
    u_xsp =np.delete(u_xsp,0,0)
    u_ysp = u_sp
    u_ysp = np.delete(u_ysp,3*Nx,0)
    u_ysp = np.delete(u_ysp,3*Nx-1,0)
    u_ysp =np.delete(u_ysp,1,0)
    u_ysp =np.delete(u_ysp,0,0)
    
    
    for r in range(1,Nx):
        u_xsp = np.delete(u_xsp,3*(Nx-r)-1,0)
        u_xsp = np.delete(u_xsp,3*(Nx-r)-2,0)
        u_ysp = np.delete(u_ysp,3*(Nx-r)-1,0)
        u_ysp = np.delete(u_ysp,3*(Nx-r)-3,0)
    
    U_xsp = np.zeros(shape=(Nx+1, 1))
    U_xsp[1:Nx] = U_xsp[1:Nx] + u_xsp
    U_ysp = np.zeros(shape=(Nx+1, 1))
    U_ysp[0] = u_sp[0]
    U_ysp[Nx] = u_sp[3*Nx-1]
    U_ysp[1:Nx] = U_ysp[1:Nx] + u_ysp


    return U_xs, U_ys, U_xsp, U_ysp

def natfreq_mode_shapes(M_s, K_s, M_sp, K_sp):
    SM = np.linalg.solve(M_s, K_s)
    SpM = np.linalg.solve(M_sp, K_sp)
    sw, sv = np.linalg.eig(SM)
    spw, spv = np.linalg.eig(SpM)
    
    wr_s = np.sqrt(sw)
    wr_sp = np.sqrt(spw)
    ms_s = sv
    ms_sp = spv
    return wr_s, wr_sp, ms_s, ms_sp

def massnormalize(ms_s, M_s, ms_sp, M_sp, size1, size2):

    ms = np.zeros(shape=(size1,1))
    msp = np.zeros(shape=(size2,1))
    Ms_s = np.zeros(shape=(size1,size1))
    Ms_sp = np.zeros(shape=(size2,size2))
    ms_sT= np.transpose(ms_s)
    ms_spT= np.transpose(ms_sp)
    for j in range(0,size1):
        ms[j] = np.dot(np.dot(ms_sT[:,j],M_s),ms_s[:,j])
        Ms_s[:,j] = 1/np.sqrt(abs(ms[j]))*ms_s[:,j]
    for j in range(0,size2):
        msp[j] = np.dot(np.dot(ms_spT[:,j],M_sp),ms_sp[:,j])
        Ms_sp[:,j] = 1/np.sqrt(abs(msp[j]))*ms_sp[:,j]   
        
    return Ms_s, Ms_sp
        
def modalForces(Ms_s, Ms_sp, F_s, F_sp):
    Ms_sT = np.transpose(Ms_s)
    Ms_spT = np.transpose(Ms_sp)
    MF_s = np.dot(Ms_sT,F_s)
    MF_sp = np.dot(Ms_spT,F_sp)
    
    return MF_s, MF_sp


# --------------------------
# MAIN

#
# Input ------------------------------------------------------
#
    
L1 = 1.0    # Lengh of beam
Nx = 40 # Number of elements (MUST BE EVEN)

# Material Properties
E = 200*10**9   
A = 0.0032258    
I = 0.0000022560
rho = 7800
px = 0
pl = 0
qy = 99.7903*9.81
ks = 100000000



    
#
# Meshing ----------------------------------------------------
#
    
Nodes, Connectivity = Mesh1D(L1, Nx)

#
# Element calculations and assembly --------------------------
#



for e in range(0, Nx):
    Nodes_el = [ Nodes[Connectivity[e,0]], Nodes[Connectivity[e,1]] ]
    M_el, K_el, F_el = LinElement1D(Nodes_el, E, I, A, rho, qy, px)


M, K, F = FullMatrix(M_el, K_el, F_el,Nx,pl) 

M_s, K_s, F_s, M_sp, K_sp, F_sp, u_s, u_sp = Constraint(M, K, F, ks)

#
# Displacements
#

U_xs, U_ys, U_xsp, U_ysp = Displacements(u_s, u_sp)


#
# Natural Frequencies and Mode Shapes
#

wr_s, wr_sp, ms_s, ms_sp = natfreq_mode_shapes(M_s, K_s, M_sp, K_sp)

#
# Modal Stiffness Matrix
#


MK_s = np.diag(wr_s)
MK_sp = np.diag(wr_sp)

wr_s.sort()
wr_sp.sort()

size1 = len(wr_s)
size2 = len(wr_sp)

Ms_s, Ms_sp = massnormalize(ms_s, M_s, ms_sp, M_sp, size1, size2)

#
# Modal Forces-----------------------
#

MF_s, MF_sp = modalForces(Ms_s, Ms_sp, F_s, F_sp)


#
# Plotting --------------------------
# 

X = np.arange(Nx+1)
X = X*L1/Nx
x = np.linspace(0,L1,1001)
# Analytical Solution ----------------------

def func(x):
    
    ua = ((qy*x**2)/(24*E*I))*((L1-x)**2)
    
    return ua

ua = func(x)

ub = ((1/(E*I))*((-1*((qy*x**4)/(24))) + ((1/12)*qy*L1*x**3)))  - ((1/24/E/I)*qy*L1**3*x)  - ((1/(2*ks))*(qy*L1))



fig1, ax1 = plt.subplots()
ax1.plot(X, U_ys*10**6, marker = 'o', color = 'r', label = 'Numerical')
ax1.plot(x, ua*10**6, color = 'b', label = 'Analytical')
plt.grid(True)

ax1.set(xlabel='Beam Location (m)', ylabel='Displacement in y direction (μm)', title='2D Linear Elastic Beam with Supports')
plt.legend()
plt.savefig('2DElastic_Supports.png', dpi = 600 , transparent = True )



fig2, ax2 = plt.subplots()
ax2.plot(X, U_ysp*10**6, marker = 'o', color = 'r', label = 'Numerical')
ax2.plot(x, -ub*10**6, color = 'b', label = 'Analytical')
plt.grid(True)
ax2.set(xlabel='Beam Location (m)', ylabel='Displacement in y direction (μm)', title='2D Linear Elastic Beam with Spring Supports')
plt.legend()

plt.savefig('2DElastic_Springs.png', dpi = 600 , transparent = True )

print(func(0.5))