# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import numpy as np
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
    print(R)
    print(alpha/beta)
    theta = np.arctan(beta/alpha)
    print(theta)
    c = np.cos(-theta)
    s = np.sin(-theta)
    R[i,i] = 1*c
    R[i,j] = -1*s
    R[j,j] = 1*c
    R[j,i] = 1*s
    print(R)
    
    
    print((c*alpha) - (s*beta))
    print((s*alpha) + (c*beta))
    print(np.sqrt(alpha**2+beta**2))
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
i=1


k = 0
alpha = testMatrix[k,k]
beta = testMatrix[k,k+1]
R = givens_rot(alpha, beta, k, k+1, n)
B = testMatrix
B = np.dot(B,R.T)
print(B)
#V = np.dot(V,R)
alpha = testMatrix[k,k]
beta = testMatrix[k+1,k]
R = givens_rot(alpha, beta, k, k+1, m)
B = np.dot(R,B)
print(B)
#U = np.dot(U,R)

#%%
def golub_kahan_svd_step(B, U, V, iLower, iUpper):

    #Arguments:
        #B -- A numpy array of size m x n. Upper diagonal matrix.
        #U -- A numpy array of size m x m. Unitary matrix.
        #V -- A numpy array of size n x n. Unitary matrix.
        #iLower, iUpper -- Identify the submatrix B22.
    #Returns:
        #B -- A numpy array of size mxn. Upper diagonal matrix with smaller values
        #on the upper diagonal elements.
        #U -- A numpy array of size m x m. Unitary matrix.
        #V -- A numpy array of size m x m. Unitary matrix.
    
        
    for k in range(iLower, iUpper):
        alpha = B[k,k]
        beta = B[k,k+1]
        R = givens_rot(alpha, beta, k, k+1, n)
        B = np.dot(B,R.T)
        print(B)
        V = np.dot(V,R)
        alpha = B[k,k]
        beta = B[k+1,k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B = np.dot(R,B)
        print(B)
        U = np.dot(U,R)
    







    