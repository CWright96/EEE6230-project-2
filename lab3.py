# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import numpy as np
#%%
def givens_rot(alpha, beta, i, j, m):

    #Arguments:
        #alpha-- The i-th component of a vector b.
        #beta -- The j-th component of a vector b.
        #i -- The index i.
        #j -- The index j.
        #m -- The length of vector b.
    #Returns:
        #R -- A numpy array of size mxm. Givens Rotation Matrix. 
        
    R = np.eye(m)                 #Identity Matrix of teh dimension m
    theta = np.arctan(beta/alpha) #Theta is obtained by Tan inverse of teh opp by adj
    c = np.cos(-theta)            #c is the cos of the rotation angle theta
    s = np.sin(-theta)            #s is the sin of the rotation angle theta
    
    #Assigning the values into respect locations in the rotation matrix
    R[i,j] = -1*s
    R[i,i] = 1*c                  
    R[j,j] = 1*c
    R[j,i] = 1*s
    
    return R                     #Returns the Rotation Matrix R
    #%%
def golub_kahan_svd_step(B, U, V, iLower, iUpper):
#    Arguments:
#        B -- A numpy array of size m x n. Upper diagonal matrix.
#        U -- A numpy array of size m x m. Unitary matrix.
#        V -- A numpy array of size n x n. Unitary matrix.
#        iLower, iUpper -- Identify the submatrix B22.
#    Returns:
#        B -- A numpy array of size mxn. Upper diagonal matrix with smaller values
#        on the upper diagonal elements.
#        U -- A numpy array of size m x m. Unitary matrix.
#        V -- A numpy array of size m x m. Unitary matrix.
    
    #1
    B22=B[iLower:iUpper, iLower:iUpper] #Creating the Diagonal Block
    tempMat = np.dot(B22.T,B22)         #A temporary matrix which is the product of Diogonal Block and its Transpose
    m,n = tempMat.shape                 #Determining the shape
    
    #2
    C = tempMat[n-2:n,n-2:n]            #Lower right 2*2 Matrix of teh block
    
    #3
    eigs,_ = np.linalg.eig(C)           #Eigen Values of C
    
    #4
    #Setting the value of mu to the closest value of c22
    if np.abs(eigs[0]-C[1,1])<np.abs(eigs[1]-C[1,1]):
        mu = eigs[0]
    else:
        mu = eigs[1]
        
    #5
    #Changing the value of alpha and beta 
    k = iLower
    alpha = B[k,k]**2-mu
    beta = B[k,k]*B[k,k+1]
    m,n = B.shape                       #Shape of B   
    
    #6
    for k in range (iLower,iUpper-1):   #Setting the range of k
        
        #Performing rotation over n
        R = givens_rot(alpha, beta, k, k+1, n) 
        B = np.dot(B,R.T)               # B is the dot product of B and Tranpose of the rotational matrix
        V = np.dot(V,R.T)               # V is the dot product of V and Tranpose of the rotational matrix
        alpha=B[k,k]                    #Setting location of alpha
        beta=B[k+1,k]                   #Setting location of Beta
        
        #Performing rotation over m
        R = givens_rot(alpha, beta, k, k+1, m)
        B=np.dot(R,B)                   # B is the dot product of B and Tranpose of the rotational matrix
        U=np.dot(U,R.T)                 # U is the dot product of U and Tranpose of the rotational matrix
        if ( k < iUpper-2 ):            
            alpha=B[k,k+1] 
            beta=B[k,k+2]

    return B, U, V                      #Returning the value of B, U and V




