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

U=np.eye(m,m)
print('U')
print(U)
V=np.eye(n,n)
print('V')
print(V)
i=1
B = np.copy(testMatrix)
iLower=B[0,0]
iUpper=n-1

#(X,Y,Z)=golub_kahan_svd_step(B, U, V, iLower, iUpper)

#%%

#
def golub_kahan_svd_step(B, U, V, iLower, iUpper):

    #1.
    
    
    print(iUpper)
    B22=B[iLower,iUpper]
    print('B22')
    print(B22)
    
    
    #2.
    C=[B22]
    
    #3.
    #eigen values
    
    #4.
    #Closer values
    
    #5.
    #K
    
    #6.
    
    for k in range (iLower,iUpper):
        alpha = testMatrix[k,k]
        beta = testMatrix[k,k+1]
        R = givens_rot(alpha, beta, k, k+1, n)
        
        B = np.dot(B,R.T)
        print(B)
        V = np.dot(V,R)
        print('newV')
        print(V)
        alpha=B[k,k]
        beta=B[k+1,k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B=np.dot(R,B)
        print('newB')
        print(B)
        U=np.dot(U,R)
        print('newU')
        print(U)
        if ( k <= iUpper-1 ):
            alpha=B[k,k+1] 
            beta=B[k+2,k]

    return(B, U, V, iLower, iUpper)
    
(X,Y,Z,xx,yy)=golub_kahan_svd_step(B, U, V, iLower, iUpper)


print(X)


    