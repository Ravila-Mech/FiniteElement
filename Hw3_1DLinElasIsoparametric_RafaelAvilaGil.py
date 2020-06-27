#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 01:17:35 2020

@author: rafaelavila
"""

#!/usr/bin/env python3
# Main - 1D elastic bar - Linear FEM

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
    # Return qudrature point positions Xiq and weights WQ for a Gauss quadrature 
    # rule in 1D of a given order
    
    if order == 1:      
        WQ = 2
        XIq = 0
    elif order == 2:       
        WQ = [1,1]
        XIq = [ (-1)/np.sqrt(3) , 1/np.sqrt(3) ]    
    else:
        print("Error: quadrature rule not implemented")
    
    return WQ, XIq
xiq = np.linspace(0,1,3)
def Shape1D(xiq, type):
#    # Return 1D shape functions N and shape functions derivatives DN evaluated at xiq
#    # Type defines the order of the 1D shape functions: linear or quadratic
#    
    if type == 1:      # Linear
        # TODO     
        N = [ 0.5*(1-xiq) , 0.5*(1+xiq) ]    
        DN =  [-N[0] , N[1] ]
#        
    elif type == 2:    # Quadratic
#        # TODO
        N =[0.5*(xiq)*(1-xiq), 1-xiq**2, 0.5*(xiq)*(1-xiq) ]
        DN = [-N[0],N[1],N[2]]             
    else:
        print("Error: type of shape function not implemented")
#    
    return N, DN
#    

def LinElement1D(Nodes_el, EA, q):
    # Generates load vector and stiffness matrix at the element level
    # Nodes_el = contains element nodal coordinates 
    # EA = element stiffness - constant on each element
    # q = distributed load (assumed constant)    
    WQ, XIq = Quadrature1D(1)
    
    N, DN = Shape1D(XIq, 1)
        
    # Compute shape functions derivatives in the X domain
        
    DXDxi = DN[0]*Nodes[0] + DN[1]*Nodes[1]
    
    
    DNX = [DN[0]*(1/DXDxi) , DN[1]*(1/DXDxi)]
    
#Stiffness matrix at element level 
    
    K_el = np.zeros(shape=(2, 2))
    
     
    el_size = Nodes_el[1] - Nodes_el[0]

#We use Gauss quadrature
    
    K_el[0,0] = (EA/(el_size))*DNX[1]*DXDxi*WQ   #DxDxi here is det(J)
    K_el[0,1] = (EA/(el_size))*DNX[0]*DXDxi*WQ  
    K_el[1,0] = (EA/(el_size))*DNX[0]*DXDxi*WQ  
    K_el[1,1] = (EA/(el_size))*DNX[1]*DXDxi*WQ  
    
    
#force vector at element level
        
    q_el = [ 0.5*q*el_size, 0.5*q*el_size ]
 
    
  
    return K_el, q_el
  
# --------------------------
# MAIN
#
# Input ------------------------------------------------------
#
    
L1 = 1.0    # Lengh of elastic bar
Nx = 10  # Number of elements

# Material Properties
EA = np.ones(shape=(Nx, 1))     
for i in range(0, Nx):      
    EA[i,0] = 1
#    EA[i,0] = (i+1)     # Modify this loop to assign different material properties per element

# EBC

#EBC = np.array([[0, 0],[Nx, 1]])    # Assign EBC in the form [dof, dof value]
EBC = np.array([[0, 0]])   

# Loads and NBC
q = 2        # Distributed load (assumed constant )
NBC = [[Nx, 1.0]]   # Assign NBC in the form [dof, load value]   
#
# Meshing ----------------------------------------------------
#
    
Nodes, Connectivity = Mesh1D(L1, Nx)

#
# Element calculations and assembly --------------------------
#

K_model = np.zeros(shape=(Nx+1, Nx+1))
f_model = np.zeros(shape=(Nx+1, 1))
for e in range(0, Nx):
    Nodes_el = [ Nodes[Connectivity[e,0]], Nodes[Connectivity[e,1]] ]
    K_el, q_el = LinElement1D(Nodes_el, EA[e], q)
    
    a = Connectivity[e,0]
    b = Connectivity[e,1]
    
    K_model[a,a] += K_el[0,0]
    K_model[a,b] += K_el[0,1]
    K_model[b,a] += K_el[1,0]
    K_model[b,b] += K_el[1,1]
    
    f_model[a] += q_el[0]
    f_model[b] += q_el[1]
        
#
# Apply element EBC and NBC --------------------------
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

f_A = f_model[FreeDOF, :]
f_B = f_model[ConstrDOF, :]

u_B = EBC[:,1]
#
# Solve for displacements and reaction forces - plot solution ----------------
# 

u_A = np.linalg.solve(K_AA, f_A - np.dot(K_AB,u_B).reshape(len(FreeDOF),1))
u   = np.zeros(shape=(Nx+1, 1))
u[FreeDOF, :]   = u_A
u[ConstrDOF, :] = u_B.reshape(len(ConstrDOF),1)

ReactionForces = -f_B + np.dot(K_BA,u_A).reshape(len(ConstrDOF),1) + np.dot(K_BB, u_B).reshape(len(ConstrDOF),1)


plt.plot(Nodes,u,'-o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.show()



print(K_el)
print(q_el)









