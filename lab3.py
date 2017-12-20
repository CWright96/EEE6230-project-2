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
    R = np.eye(m)
    theta = np.arctan(beta/alpha)
    c = np.cos(-theta)
    s = np.sin(-theta)
    R[i,i] = 1*c
    R[i,j] = -1*s
    R[j,j] = 1*c
    R[j,i] = 1*s
    return R    






#%%
b=np.array([[1.0],[2.0],[3.0],[4.0],[5.0]])
print(b)
    
m = b.shape[0]
i=1
j=2
alpha = b[i]
beta = b[j]


    
R = givens_rot(alpha, beta, i, j, m)

Rdotb = np.dot(R,b)
print(Rdotb)
#%%



    #%%

#
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

    
    
    
    B22=B[iLower:iUpper, iLower:iUpper]
    print('B22')
    print(B22)
    
    tempMat = np.dot(B22.T,B[2,2])
    m,n = tempMat.shape
    C = tempMat[n-2:n,n-2:n]
    
    eigs,_ = np.linalg.eig(C)
    
    if np.abs(eigs[0]-C[1,1])<np.abs(eigs[1]-C[1,1]):
        mu = eigs[0]
    else:
        mu = eigs[1]
    
    k = iLower
    alpha = B[k,k]**2-mu
    beta = B[k,k]*B[k,k+1]
    
    
    
    
    for k in range (iLower,iUpper):

        R = givens_rot(alpha, beta, k, k+1, n)
        
        B = np.dot(B,R.T)
        print(B)
        V = np.dot(V,R.T)
        print('newV')
        print(V)
        alpha=B[k,k]
        beta=B[k+1,k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B=np.dot(R,B)
        print('newB')
        print(B)
        U=np.dot(U,R.T)
        print('newU')
        print(U)
        if ( k <= iUpper-1 ):
            alpha=B[k,k+1] 
            beta=B[k,k+2]

    return(B, U, V, iLower, iUpper)


#%%
testMatrix = np.array([[1,5,0,0],
                       [0,2,6,0],
                       [0,0,3,7],
                       [0,0,0,4],
                       [0,0,0,0],
                       [0,0,0,0]])
print(testMatrix)
print(testMatrix.shape)
m = testMatrix.shape[0]
n = testMatrix.shape[1]

U=np.eye(m,m)
print('U')
print(U)
V=np.eye(n,n)
print('V')
print(V)
i=1
B = np.copy(testMatrix)
iLower=1
iUpper=4

(X,Y,Z)=golub_kahan_svd_step(B, U, V, iLower, iUpper)    


    