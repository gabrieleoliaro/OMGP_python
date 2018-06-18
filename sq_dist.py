#!/usr/bin/python

import numpy as np

def sq_distONE(a):
    """
    sq_dist - a function to compute a matrix of all pairwise squared distances
    between two sets of vectors, stored in the columns of the two matrices, a
    (of size D by n) and b (of size D by m). If only a single argument is given
    or the second matrix is empty, the missing matrix is taken to be identical
    to the first.

    Usage: C = sq_dist(a, b)
       or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])
    
    Where a is of size Dxn, b is of size Dxm (or empty), C is of size nxm.
    
    Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-12-13.

    """
    [D, n] = a.shape
    mu = np.mean(a, axis=1)
    a = a - mu                              # Subtract off mean
    b = a
    m = n

def sq_distTWO(a, b):
    """
    sq_dist - a function to compute a matrix of all pairwise squared distances
    between two sets of vectors, stored in the columns of the two matrices, a
    (of size D by n) and b (of size D by m). If only a single argument is given
    or the second matrix is empty, the missing matrix is taken to be identical
    to the first.

    Usage: C = sq_dist(a, b)
       or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])
    
    Where a is of size Dxn, b is of size Dxm (or empty), C is of size nxm.
    
    Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-12-13.

    """
    if b == []:
        return sq_distONE(a)

    [D, n] = a.shape
    [d, m] = b.shape

    if d != D:
        raise ValueError("Error: column lengths must agree.")

    mu = (m / (n + m)) * mean(b, axis=1) + (n / (n + m)) * mean(a, axis=1)
    a = a - mu
    b = b - mu

    C = np.sum(a * a, 0).conj().transpose() + (np.sum(b * b, 0) - 2 * a.conj().transpose() * b)
    C = np.maximum(C,0)                 # Numerical noise can cause C to negative i.e. C > -1e-14
        
    
    
