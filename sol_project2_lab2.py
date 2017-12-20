#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
% EEE6230- Scientific Software Development for Biomedical Imaging
% Project 2:: Programming Lab 2
"""
#%% import required libraries
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

#%% Follow-up work: part 1
def test_bidiag_reduction(fun, A, eps = 1e-8):
    '''
    Arguments:
        fun -- The function handle 
        A   -- A numpy array of size mxn. Data matrix
        eps -- A small value used as a threshold to check
               if a value is zero or not
        
    Returns:
        True -- If the code is correct
        False -- If the code is not correct
    '''
    B, U, V = fun(A)
    
    # check if A = U*B*V^T
    newA = np.dot(np.dot(U,B), V.T)
    if np.sum(np.abs(A-newA))> eps:
        return False
    
    # check if U*U^T = I
    if np.sum(np.abs(np.eye(A.shape[0])-np.dot(U,U.T)))> eps:
        return False
    
    # check if V*V^T = I
    if np.sum(np.abs(np.eye(A.shape[1])-np.dot(V,V.T)))> eps:
        return False
    
    # check if B is upper bidiagonal
    if np.abs(np.sum(B) - np.sum(np.diag(B,0)) - np.sum(np.diag(B,1))) > eps:
        return False
    
    # otherwise
    return True
    




   