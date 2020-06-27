#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:08:47 2020

@author: rafaelavila
"""

import matplotlib.pyplot as plt
import numpy as np
# --------------------------
# Functions definition
def Mesh1D(L1, Nx):
    
   Nodes = np.linspace(0 , L1 , Nx+1)  
   Connectivity =  np.zeros((Nx,2))          

   return Nodes, Connectivity
        
def LinElement1D(Nodes_el, EA, q):
    #"Element"   
    l = Nodes_el  
   #"Local Stiffness"    
    K_el = ((EA)/l)*np.array([[1, -1], [-1, 1]])   
   #"Load Vector" 
    q_el = -1*np.array([[q*l*0.5], [q*l*0.5]])        
    return K_el, q_el
# 
# --------------------------
# MAIN
#
# Input ------------------------------------------------------
#    
L1 = 3.0    # Lengh of elastic bar
Nx = 4# Number of elements
# Material Properties
EA = np.ones(shape=(Nx, 1))     
for i in range(0, Nx):      # Modify this loop to assign different material properties per element
    EA[i,0] = 1    
# EBC
EBC = np.array([[0, 0]])    # Assign EBC in the form [dof, dof value]  

# Distributed loads and NBC
q = 1
    # Distributed load (assumed constant)
    

NBC = [Nx, 1]   # Assign NBC in the form [dof, load value] 
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
    
    Connectivity[e , :]= [e, e+1]
    
    Nodes_el = Nodes[np.int(Connectivity[e,1])] - Nodes[np.int(Connectivity[e,0])]
       
    K_el, q_el = LinElement1D(Nodes_el, EA[i], q)
       
    a = np.int(Connectivity[e,0])
    b = np.int(Connectivity[e,1])    
# K matrix assembly
    K_model[a,a] += K_el[0,0]
    K_model[a,b] += K_el[0,1]
    K_model[b,a] += K_el[1,0]
    K_model[b,b] += K_el[1,1]
    
    #K_model[e:e + 2, e:e + 2] += K_el  <---- Alternative for Assembly
    
 # F vector Assembly   
    f_model[a] += -q_el[0]   
    f_model[b] += -q_el[1] 
    
    #f_model[e:e+2] += -q_el           <-----  Alternative for Assembly  
       
# Apply element EBC --------------------------
#      
K_model[:,EBC[0,0]] = 0       #<---- Control this form the inputs
K_model[EBC[0,0],:] = 0
    
        # Zero first row and first column
#K_model[0, :] = 0.0              <----Alternative with limitations-less generalized
#K_model[:, 0] = 0.0
#        # Place one 
K_model[EBC[0,0], EBC[0,0]] = 1.0
#K_model[0, 0] = 1.0
#Forces
##
f_model[EBC[0,0]] = EBC[0,1]
##NBC
f_model[NBC[0]]+= NBC[1]
#----------------
#
#
#
# Solve for displacements and reaction forces - plot solution ----------------
# 
#Nodes displacement------
u = np.linalg.solve(K_model, f_model)
#forces----- 
print(f_model)
print(f_model[EBC[0,0]]) #reaction where EBC is applied
#Displacement plot----------------------
plt.ylabel('Axial Displacement  $u$')
plt.xlabel('position along the length of the bar (m)')
plt.title('Finite element solution for the 1D elastic bar Homework 1')
plt.plot(Nodes, u, 'ro-'); 
 

