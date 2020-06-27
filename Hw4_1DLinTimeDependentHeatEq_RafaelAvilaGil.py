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
    
    N, DN = Shape1D(XIq[0],1)
    
    # Compute shape functions derivatives in the X domain
    DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1]
    DNX  = [DN[0]/DXDxi, DN[1]/DXDxi]
    
    # Compute conductivity matrix
    K_el = np.zeros(shape=(2, 2))
    
    K_el[0,0] = k * DNX[0]*DNX[0] * DXDxi * WQ[0]
    K_el[0,1] = k * DNX[0]*DNX[1] * DXDxi * WQ[0]
    K_el[1,0] = k * DNX[1]*DNX[0] * DXDxi * WQ[0]
    K_el[1,1] = k * DNX[1]*DNX[1] * DXDxi * WQ[0]
    
    # Compute capacity matrix
    WQ, XIq = Quadrature1D(2)
    
    M_el = np.zeros(shape=(2, 2))
    
    for q in range(0, np.size(WQ)):
        
        N, DN = Shape1D(XIq[q],1)
        
        DXDxi = DN[0]*Nodes_el[0] + DN[1]*Nodes_el[1]
    
        M_el[0,0] += c*N[0]*N[0]*DXDxi*WQ[q]
        M_el[0,1] += c*N[0]*N[1]*DXDxi*WQ[q]
        M_el[1,0] += c*N[1]*N[0]*DXDxi*WQ[q]
        M_el[1,1] += c*N[1]*N[1]*DXDxi*WQ[q]
        
        # c = capacity*density
        
    return M_el, K_el
    
    
# --------------------------
# MAIN
#
# Input ------------------------------------------------------
#
    
L1 = 1.0    # Lengh domain
Nx = 4    # Number of elements

# Material Properties
k = 1.0     # conductance
c = 1.0     # capacity*density

# EBC
# EBC = np.array([[0, 0]]) 
# EBC = np.array([[0, 0],[Nx, 1]])    # Assign EBC in the form [dof, dof value]
EBC = np.array([[0, 0]])

   
# NBC
NBC = [[Nx, 1.0]]   # Assign NBC in the form [dof, load value]
    
    
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
    M_el, K_el = LinElement1D(Nodes_el, c, k)


print(K_el)
print(M_el)
    
















