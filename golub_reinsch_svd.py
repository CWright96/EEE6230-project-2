#%%
import numpy as np
#%% part 1: Householder reduction to bidiagonal form
def householder_transformation(b, k):
    '''
    Arguments:
        b -- A numpy array of size mx1
        k -- An integer index between 1 and m-1.
             The algorithm will set the entries below index k to zero.
        
    Returns:
        H -- A numpy array of size mxm: The Householder Matrix
    '''
    # uncomment this line if you index the first matrix element as 1
    # k = k - 1
    
    m = b.shape[0]
    w = b.copy()
    w[0:k,0] = 0
    
    s = np.sqrt(np.sum(w**2))
    
    if b[k,0]>=0:
        s = -s;
        
    w[k,0] = w[k,0] - s    
    w = w/np.sqrt(np.sum(w**2))
    H = np.eye(m,m) - 2*np.dot(w, w.T)
    return H

#%% part 2: Compute the bidiangonal reduction of matrix
def bidiag_reduction(A):
    '''
    Compute the bidiagonal reduction of matrix A = UBV.T
    Arguments:
        A -- A numpy array of size mxn
        
    Returens:
        B -- A numpy array of size mxn. Upper bidiagonal matrix.
        U -- A numpy array of size mxm. Unitary matrix.
        V -- A numpy array of size nxn. Unitary matrix.
    '''
    m = A.shape[0] # number of rows
    n = A.shape[1] # number of columns
    
    B = A.copy()
    U = np.eye(m, m)
    V = np.eye(n, n)
    
    for k in range(n):
        v = np.zeros((m, 1))
        v[0:m, 0] = B[0:m, k]
        Q = householder_transformation(v, k)
        
        B = np.dot(Q, B)
        U = np.dot(U, Q)
        
        if (k < n-2):
            v = np.zeros((n, 1))
            v[0:n, 0] = B[k, 0:n]
            P = householder_transformation(v, k+1)
            B = np.dot(B, P)
            V = np.dot(V, P)
    
    return B, U, V
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
    
    tempMat = np.dot(B22.T,B22)
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
    
    m,n = B.shape    
    
    for k in range (iLower,iUpper-1):
        R = givens_rot(alpha, beta, k, k+1, n)
        B = np.dot(B,R.T)
        print("newB")
        print(B)
        V = np.dot(V,R.T)
        alpha=B[k,k]
        beta=B[k+1,k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B=np.dot(R,B)
        print('newB')
        print(B)
        U=np.dot(U,R.T)
        if ( k < iUpper-2 ):
            alpha=B[k,k+1] 
            beta=B[k,k+2]

    return B, U, V

#%%
def golub_reinsch_svd(A, MAX_ITR = 50, eps = 1.e-8):

#    Arguments:
#        A -- A numpy array of size m x n with m >= n.
#        MAX_ITR -- Number of maximum iterations.
#        eps -- A small value used as a threshold to
#        check if a value is zero or not.
#    Returns:
#        S -- A numpy array of size n x n.
#        U -- Unitary matrix of size m x m.
#        V -- Unitary matrix of size n x n.
#        counter -- Number of iterations before convergence.
        B, U, V = bidiag_reduction(A)
        m,n = B.shape
        counter = 0
        q=0
        p=0
        while counter < MAX_ITR:
            
            print(counter)
            for i in range (0,n-1):
                if (np.abs(B[i,i+1]))<=(eps*np.abs(B[i,i]+B[i+1,i+1])):
                    B[i,i+1] = 0                    
            B33=B[n-2-q:n-q, n-2-q:n-q]
            print("B33")
            print(B33)
            #gets the values of the opposite diagonal, if they add to less than eps that block is reduced
            if np.sum(np.diag(np.fliplr(B33))) < eps:
                print("q")
                print(q)
                q+=1 #increment q so that the block being worked on is smaller
            if q == n-1:
                S = np.diag(B)
                print(np.dot(np.dot(U, B),V.T))
                print(np.dot(U,U.T))
                print(np.dot(V,V.T))
                return S, U, V, counter
            B,U,V = golub_kahan_svd_step(B,U,V,p,n-q)
                
            counter +=1       
        
        
