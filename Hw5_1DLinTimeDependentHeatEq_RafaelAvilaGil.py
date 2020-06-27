#!/usr/bin/env python3
# Main - 1D time dependent heat diffusion - Linear FEM

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
    else:
        print("Error: quadrature rule not implemented")
    
    return WQ, XIq



def Shape1D(xiq, type):
    
    if type == 1:
        N = [0.5*(1.0 - xiq), 0.5*(1.0 + xiq)]; 
        
        DN = [-0.5, 0.5];

    elif type == 2:
        
        N = [0.5*xiq*(xiq-1), (1 - xiq*xiq) , 0.5*xiq*(xiq+1)];
               
        DN = [xiq - 0.5, -2.0*xiq , xiq + 0.5];
               
    else:
        print("Error: type of shape function not implemented")
    
    return N, DN
    


def LinElement1D(Nodes_el, c, k):
    # Generates capacity and conductivity matrices at the element level
    # Nodes_el = contains element nodal coordinates 
    # c = capacity*density
    # k = conductivity 
    
    WQ, XIq = Quadrature1D(1)
    
    N, DN = Shape1D(XIq[0], 1)
    
    # Compute shape functions derivatives in the X domain
    DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1];
    DNDX = [DN[0]/DXDxi, DN[1]/DXDxi];
    
    # Compute conductivity matrix
    K_el = np.zeros(shape=(2, 2))
    
    K_el[0,0] = k * DNDX[0]*DNDX[0] * DXDxi * WQ[0];
    K_el[0,1] = k * DNDX[0]*DNDX[1] * DXDxi * WQ[0];
    K_el[1,0] = k * DNDX[1]*DNDX[0] * DXDxi * WQ[0];
    K_el[1,1] = k * DNDX[1]*DNDX[1] * DXDxi * WQ[0];
    
    # Compute capacity matrix
    WQ, XIq = Quadrature1D(2)
    
    M_el = np.zeros(shape=(2, 2))
    
    for q in range(0, np.size(WQ)):
        N, DN = Shape1D(XIq[q], 1)
        
        DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1];
    
        M_el[0,0] += c * N[0]*N[0] * DXDxi * WQ[q];
        M_el[0,1] += c * N[0]*N[1] * DXDxi * WQ[q];
        M_el[1,0] += c * N[1]*N[0] * DXDxi * WQ[q];
        M_el[1,1] += c * N[1]*N[1] * DXDxi * WQ[q];
    
    
    # Implement the lumped capacity matrix using the row sum technique
    M_el_lumped = np.zeros(shape=(2, 2))
    
    M_el_lumped[0,0] += M_el[0,0] + M_el [0,1] #Here we sum the row components of the non lumped case and place in diagonal
    M_el_lumped[1,1] += M_el[1,0] + M_el [1,1]

    
    return M_el, K_el, M_el_lumped
    
    
# -------------------------------------------
# MAIN

#
# Input ------------------------------------------------------
#
    
L1 = 1.0    # Lengh domain
Nx = 1    # Number of elements

# Material Properties
k = 1.0     # conductance
c = 2.0     # capacity*density

# EBC
# EBC = np.array([[0, 0]]) 
EBC = np.array([[0, 0],[Nx, 1]])    # Assign EBC in the form [dof, dof value]
# EBC = np.array([[0, 0]])

# NBC
NBC = []
# NBC = [[Nx, 1.0]]   # Assign NBC in the form [dof, load value]
    
# Solver parameters
Delta_t =  100.00;
t_max   = 1000;
alpha   = .0;
    
#
# Meshing ----------------------------------------------------
#
    
Nodes, Connectivity = Mesh1D(L1, Nx)

#
# Element calculations and assembly --------------------------
#

K_model = np.zeros(shape=(Nx+1, Nx+1))
M_model = np.zeros(shape=(Nx+1, Nx+1))
f_model = np.zeros(shape=(Nx+1, 1))

for e in range(0, Nx):
    Nodes_el = [ Nodes[Connectivity[e,0]], Nodes[Connectivity[e,1]] ]
    M_el, K_el, M_el_lumped = LinElement1D(Nodes_el, c, k)
    
    a = Connectivity[e,0]
    b = Connectivity[e,1]
    
    K_model[a,a] += K_el[0,0]
    K_model[a,b] += K_el[0,1]
    K_model[b,a] += K_el[1,0]
    K_model[b,b] += K_el[1,1]
    
    # By commenting/uncommenting the next lines you can switch between the
    # full and lumped capacity matrix
#    M_model[a,a] += M_el[0,0]
#    M_model[a,b] += M_el[0,1]
#    M_model[b,a] += M_el[1,0]
#    M_model[b,b] += M_el[1,1]
#    
    M_model[a,a] += M_el_lumped[0,0]
    M_model[b,b] += M_el_lumped[1,1]
    
    
#
# Apply EBC and NBC --------------------------
#      

# NBC 
for i in range( 0, np.size(NBC,0) ):      
    f_model[NBC[i][0]] += NBC[i][1]
    
dof       = np.linspace(0, Nx, Nx+1, dtype=int)
ConstrDOF = EBC[:,0].astype(int)
FreeDOF   = np.delete(dof, ConstrDOF, 0)

K_AA = K_model[FreeDOF, :][:, FreeDOF]
K_AB = K_model[FreeDOF, :][:, ConstrDOF]
K_BA = K_model[ConstrDOF, :][:, FreeDOF]
K_BB = K_model[ConstrDOF, :][:, ConstrDOF]

M_AA = M_model[FreeDOF, :][:, FreeDOF]
M_AB = M_model[FreeDOF, :][:, ConstrDOF]
M_BA = M_model[ConstrDOF, :][:, FreeDOF]
M_BB = M_model[ConstrDOF, :][:, ConstrDOF]

f_A = f_model[FreeDOF, :]
f_B = f_model[ConstrDOF, :]

T_B = EBC[:,1]



#
# Solver and plot solution
# 
# Compute v_n and n=0
d_n = np.zeros(shape=(len(FreeDOF),1)); # Initial condition
F = f_A - np.dot(K_AB,T_B).reshape(len(FreeDOF),1)
v_n = np.linalg.solve(M_AA, F)
                       

# Solve per every time step
t_n = Delta_t;
t_plot = 0.0
while (t_n <= t_max):
    print('%.3f' % t_n)
    
    # Compute predictor
    d_tilda_n1 = d_n + (1-alpha) * Delta_t * v_n
    
    # Compute F_(n+1)
    F_n1 = f_A - np.dot(K_AB,T_B).reshape(len(FreeDOF),1)
    # Solve for v_(n+1)
    coeff = alpha*Delta_t
    v_n1 = np.linalg.solve( (M_AA + coeff*K_AA) , (F_n1 - np.dot(K_AA,d_tilda_n1).reshape(len(FreeDOF),1)) )
    
    
    # Compute v_(n+1)
    d_n1 = d_tilda_n1 + coeff*v_n1
    
    # Plot solution
    if ( t_n >= t_plot ):
        T = np.zeros(shape=(Nx+1, 1))
        T[FreeDOF, :]   = d_n1
        T[ConstrDOF, :] = T_B.reshape(len(ConstrDOF),1)
        plt.plot(Nodes,T,'-o')
        plt.xlabel('x')
        plt.ylabel('T(x)')
        plt.show()
        t_plot += 0.1
    
    # Advance in time
    d_n = d_n1
    v_n = v_n1
    t_n += Delta_t
    
#print(M_model)






