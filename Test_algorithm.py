#%%
import numpy as np
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

    B, U, V, counter = fun(A)
    # check if A = U*B*V^T
    NewA = np.dot(U,B)
    NewA = np.dot(NewA,V.T)
    
    if np.sum(np.abs(A-NewA))>eps:
        print("product test failed!")
        return False

    # check if U*U^T = I
    
    # check if V*V^T = I
    
    # check if B is diagonal with positive elements 