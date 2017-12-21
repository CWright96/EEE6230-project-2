'''
    Group 1a project 2 lab 3 final code
'''
#%%
'''
    Code from previous lab sessions
'''
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
'''end of code from previous labs'''
#%%
'''
    prelab
'''   
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
#%%
'''
    lab 3
'''

'''ex 8 pt 1'''
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
'''ex 8 pt 2'''
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
        #print("newB")
        #print(B)
        V = np.dot(V,R.T)
        alpha=B[k,k]
        beta=B[k+1,k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B=np.dot(R,B)
        #print('newB')
        #print(B)
        U=np.dot(U,R.T)
        if ( k < iUpper-2 ):
            alpha=B[k,k+1] 
            beta=B[k,k+2]

    return B, U, V
#%%
'''ex 8 pt 3'''
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
            
            #print(counter)
            for i in range (0,n-1):
                if (np.abs(B[i,i+1]))<=(eps*np.abs(B[i,i]+B[i+1,i+1])):
                    B[i,i+1] = 0                    
            B33=B[n-2-q:n-q, n-2-q:n-q]
            #gets the values of the opposite diagonal, if they add to less than eps that block is reduced
            if np.sum(np.diag(np.fliplr(B33))) < eps:
                q+=1 #increment q so that the block being worked on is smaller
            if q == n-1:
                S = np.diag(B)
                #print(B)
                return S, U, V, counter
            B,U,V = golub_kahan_svd_step(B,U,V,p,n-q)
                
            counter +=1
#%%
'''ex 9 pt 1'''
#%%
def test_golub_reinsch_svd(fun, A, eps = 1e-8):

    #Arguments:
        #fun -- The function handle.
        #A -- A numpy array of size m x n. Data matrix.
        #eps -- A small value used as a threshold to check
        #if a value is zero or not.
    #Returns:
        #True -- If the code is correct.
        #False -- If the code is not correct.

    S, U, V, counter = fun(A)
    # check if A = U*B*V^T
    #print(S)
    Um = U.shape[0]
    Vn = V.shape[1]
    a = np.zeros((Um,Vn),int)
    np.fill_diagonal(a,S)
    B = a
    NewA = np.dot(U,B)
    NewA = np.dot(NewA,V.T)
    
    if np.sum(np.abs(A-NewA))<eps:
        print("product test failed!")
        return False

    # check if U*U^T = I
    if np.sum(np.abs(np.eye(A.shape[0])-np.dot(U,U.T)))> eps:
        print("U.U.T identity test Failed!")
        return False
    
    # check if V*V^T = I
    if np.sum(np.abs(np.eye(A.shape[1])-np.dot(V,V.T)))> eps:
        print("V.V.T identity test Failed!")
        return False    
    
    # check if B is diagonal with positive elements
    
    B_copy = np.array(B, copy = True)#create a copy of the B Matrix
    np.fill_diagonal(B_copy,0)
    for i in np.nditer(B_copy): #iterate over the array    
        #print(i)        
        if np.absolute(i) > eps:#the value i is an element in the array
            print("Diagonal test failed")
            return False     #if i is greater than eps
            break               #the matrix is not diagonal, exit loop
    #if the test makes it this far the matrix is diagonal
    for i in np.nditer(np.diagonal(B)):
        if i < 0:
            print("element is not positive")
            return False
    #all tests passed
    
    return True
#%%
''' ex 9 pt 2'''
#%%
def mySVD(A):
    m,n = A.shape
    print("The matrix A has dimentions of: " + str(m)+ " by " + str(n))
    
    if m>=n:
        S,U,V,count = golub_reinsch_svd(A)
    else:
        S,V,U,count = golub_reinsch_svd(A.T)   
    return S,U,V
#%%
''' ex 9 pt 3'''
#%%
    #compare MySVD to numpy SVD
    
    #Execution time
def comparison(A):
    
    import timeit
    start_time = timeit.default_timer()
    S,U,V = mySVD(A)
    elapsed1 = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    SP,UP,VP = np.linalg.svd(A)  #"P" stands for python
    elapsed2 = timeit.default_timer() - start_time
    return elapsed1,elapsed2    