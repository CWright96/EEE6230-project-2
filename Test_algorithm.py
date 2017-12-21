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
    Um,Un = U.shape
    a = np.zeros((Um,Un),float)
    B = np.fill_diagonal(a,B)
    NewA = np.dot(U,B)
    NewA = np.dot(NewA,V.T)
    
    if np.sum(np.abs(A-NewA))>eps:
        print("product test failed!")
        return False

    # check if U*U^T = I
    if np.sum(np.abs(np.eye(A.shape[1])-np.dot(U,U.T)))> eps:
        print("U.U.T identity test Failed!")
        return False
    
    # check if V*V^T = I
    if np.sum(np.abs(np.eye(A.shape[1])-np.dot(V,V.T)))> eps:
        print("V.V.T identity test Failed!")
        return False    
    
    # check if B is diagonal with positive elements
    
    B_copy = np.array(B, copy = True)#create a copy of the B Matrix
            
    B_copy = (B_copy - np.diag(np.diagonal(B_copy)))#remove the diagonal values
    #all other values in the matrix should be 0

    for i in np.nditer(B_copy): #iterate over the array            
        if np.absolute(i) > eps:#the value i is an element in the array
            print("Diagonal test failed")
            return False     #if i is greater than eps
            break               #the matrix is not diagonal, exit loop
    #if the test makes it this far the matrix is diagonal
    for i in np.nditer(np.diagonal(B_copy)):
        if i < 0:
            print("element is not positive")
            return False
    #all tests passed
    
    return True
  