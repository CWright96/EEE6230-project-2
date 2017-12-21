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
    
#Part 1   
A = np.random.randn(10,5)*10.0  #Create a Random Matrix of given dimension
B,U,V = bidiag_reduction(A)     #Perform Bidiagonal Reduction 

U1,S1,V1 = np.linalg.svd(A)     #Singular Values of A
U2,S2,V2 = np.linalg.svd(B)     #Singular values of B

Utest = U1-U2                   # Comparing the U values
Stest = S1-S2                   #Comparing the Sigma Values
Vtest = (V1.T-V2.T)             #Compring the V values

#Part2
NewU = np.dot(U,U2)             #Obtaining the new unitary Matrix U
NewV = np.dot(V,V2)             #Obtaining the new unitary Matrix V
NewSigma= S2                    #Sigma values remain the same



