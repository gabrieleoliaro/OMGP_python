#!/usr/bin/python
from __future__ import division
import numpy as np
from omgpbound import *

def minimize(loghyper, f, length, learn, covfunc, M, X, Y):
    """
    Minimize a differentiable multivariate function using conjugate gradients.
    

    Usage: [loghyper, fX] = minimize(loghyper, f, length, learn, covfunc, M, X, Y)
    loghyper    initial guess; may be of any type, including struct and cell array
    f           the name of the function to be minimized. The function
                f must return two arguments, the value of the function, and it's
                partial derivatives wrt the elements of loghyper. The partial derivative
                must have the same type as loghyper.
    length      length of the run; if it is positive, it gives the maximum number of
                line searches, if negative its absolute gives the maximum allowed
                number of function evaluations. Optionally, length can have a second
                component, which will indicate the reduction in function value to be
                expected in the first line-search (defaults to 1.0).
    learn, covfunc, M, X, Y parameters are passed to the function f.

    loghyper    the returned solution
    fX          vector of function values indicating progress made

    The function returns when either its length is up, or if no further progress
    can be made (ie, we are at a (local) minimum, or so close that due to
    numerical problems, we cannot get any closer). NOTE: If the function
    terminates within a few iterations, it could be an indication that the
    function values and derivatives are not consistent (ie, there may be a bug in
    the implementation of your "f" function).

    The Polack-Ribiere flavour of conjugate gradients is used to compute search
    directions, and a line search using quadratic and cubic polynomial
    approximations and the Wolfe-Powell stopping criteria is used together with
    the slope ratio method for guessing initial step sizes. Additionally a bunch
    of checks are made to make sure that exploration is taking place and that
    extrapolation will not be unboundedly large.

    Copyright (C) 2001 - 2010 by Carl Edward Rasmussen, 2010-01-03
    """

    # SIG and RHO are the constants controlling the Wolfe-
    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the initial
    # point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with RHO.

    # The code falls naturally into 3 parts, after the initial line search is
    # started in the direction of steepest descent. 1) we first enter a while loop
    # which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
    # have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
    # enter the second loop which takes p2, p3 and p4 chooses the subinterval
    # containing a (local) minimum, and interpolates it, unil an acceptable point
    # is found (Wolfe-Powell conditions). Note, that points are always maintained
    # in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
    # conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
    # was a problem in the previous line-search. Return the best value so far, if
    # two consecutive line-searches fail, or whenever we run out of function
    # evaluations or line-searches. During extrapolation, the "f" function may fail
    # either with an error or returning Nan or Inf, and minimize should handle this
    # gracefully.

    INT = 0.1               # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0               # extrapolate maximum 3 times the current step-size
    MAX = 20                # max 20 function evaluations per line search
    RATIO = 10              # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2

    length = np.matrix((length))
    
    if max(length.shape) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1
        
    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation' 


    i = 0                                                               # Run length counter, initialized to zero
    ls_failed = 0                                                       # No previous line search has failed
    if f == 'omgpbound':
        [f0, df0] = omgpboundA(loghyper, learn, covfunc, M, X, Y)       # Get function value and gradient
        
    Z = loghyper

    print('%s %6i;  Value %4.6e' % (S, i, f0))
    fX = np.matrix([f0])
    i = i + (length<0)                                                  # count epochs?!

    s = -df0
    d0 = np.matmul(-s.conj().transpose(), s)                            # initial search direction (steepest) and slope
    x3 = np.divide(red, (np.subtract(1, d0)))                                                 # initial step is red/(|s|+1)

    while i < abs(length):                                              # while not finished
        i = i + (length > 0)                                            # count iterations?!

    # Make a copy of current values
        X0 = loghyper
        F0 = f0
        dF0 = df0
        if length > 0:
            m = MAX
        else:
            m = min(MAX, -length - i)

        while (True):                                                       # keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while not success and m > 0:
                try:
                    m = m - 1
                    i = i + (length < 0)                                    # count epochs?!
                    if f == 'omgpbound':
                        [f3, df3] = omgpboundA(np.add(loghyper, np.multiply(x3, s)), learn, covfunc, M, X, Y)
                        
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise Warning('')
                    success = 1
                except:                                                     # catch any error which occured in f
                    x3 = np.divide((x2 + x3), 2)                                      # bisect and try again
    
            if f3 < F0:                                                     # keep best values
                X0 = np.add(loghyper, np.multiply(x3, s))
                F0 = f3
                dF0 = df3
            d3 = np.matmul(df3.conj().transpose(), s)                      # new slope
            if d3 > np.multiply(SIG, d0) or f3 > np.add(f0, np.multiply(np.multiply(x3, RHO), d0)) or m == 0:          # are we done extrapolating?
                break 
      
            # Move point 2 to point 1
            x1 = x2
            f1 = f2
            d1 = d2

            # Move point 3 to point 2
            x2 = x3
            f2 = f3
            d2 = d3

            # Make cubic extrapolation
            A = np.add(np.multiply(6, np.subtract(f1, f2)), np.multiply(np.multiply(3, np.add(d2, d1)), np.subtract(x2, x1)))
            B = np.subtract(np.multiply(3, np.subtract(f2, f1)), np.multiply(np.add(np.multiply(2, d1), d2), np.subtract(x2, x1)))
            

            Z = B + np.sqrt(complex(np.subtract(np.multiply(B, B), np.multiply(np.multiply(A, d1), np.subtract(x2, x1)))))
            if Z != 0.0:
                x3 = np.subtract(x1, np.divide(np.multiply(d1, np.power(np.subtract(x2, x1), 2)), Z))              # num. error possible, ok!
            else: 
                x3 = np.inf

            
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:            # num prob | wrong sign?
                x3 = np.multiply(x2, EXT)                                                       # extrapolate maximum amount
            elif x3 > np.multiply(x2, EXT):                                             # new point beyond extrapolation limit?
                x3 = np.multiply(x2, EXT)                                              # extrapolate maximum amount
            elif x3 < np.add(x2, np.multiply(INT, np.subtract(x2, x1))):                                 # new point too close to previous point?
                x3 = np.add(x2, np.multiply(INT, np.subtract(x2, x1)))
            x3 = np.real(x3)

        while (np.abs(d3) > - np.multiply(SIG, d0) or f3 > np.add(f0, np.multiply(x3, np.multiply(RHO, d0)))) and m > 0:  # keep interpolating
            if d3 > 0 or f3 > np.add(f0, np.multiply(x3, np.multiply(RHO, d0))):                           # choose subinterval
                # Move point 3 to point 4
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                # Move point 3 to point 2
                x2 = x3
                f2 = f3
                d2 = d3
    
            if f4 > f0:
                x3 = np.subtract(x2, np.divide((np.multiply(0.5, np.multiply(d2, np.power(np.subtract(x4, x2), 2)))), (np.subtract(np.subtract(f4, f2), np.multiply(d2, np.subtract(x4, x2))))))    # quadratic interpolation
            else:
                A = np.add(np.divide(np.multiply(6, np.subtract(f2, f4)), np.subtract(x4, x2)), np.multiply(3, np.add(d4, d2)))                        # cubic interpolation
                B = np.subtract(np.multiply(3, np.subtract(f4, f2)), np.multiply((np.add(np.multiply(2,d2), d4)), np.subtract(x4, x2)))
                if A != 0:
                    x3 = np.add(x2, np.divide(np.subtract(np.sqrt(np.subtract(np.multiply(B,B), np.multiply(A, np.multiply(d2, np.power(np.subtract(x4,x2), 2))))), B), A))
                                                     # num. error possible, ok!
                else:
                    x3 = np.inf
    
            if np.isnan(x3) or np.isinf(x3):
                x3 = np.divide(np.add(x2, x4), 2)                                          # if we had a numerical problem then bisect
            x3 = max(min(x3, np.subtract(x4, np.multiply(INT, np.subtract(x4, x2)))), np.add(x2, np.multiply(INT, np.subtract(x4, x2))))   # don't accept too close
            [f3, df3] = omgpboundA((loghyper + np.multiply(x3, s)), learn, covfunc, M, X, Y)
            if f3 < F0:
                # keep best values
                X0 = np.add(loghyper, np.multiply(x3, s))
                F0 = f3
                dF0 = df3 
            m = m - 1
            i = i + (length < 0)                                            # count epochs?!
            d3 = np.matmul(df3.conj().transpose(), s)                       # new slope
        # end interpolation

        
        if (np.abs(d3) < -np.multiply(SIG, d0) and f3 < np.add(f0, np.multiply(x3, np.multiply(RHO, d0)))):                 # if line search succeeded
            loghyper = np.add(loghyper, np.multiply(x3, s))
            f0 = f3
            fX = np.insert(fX, fX.shape[0], f0,axis=0)                                           # update variables
            print('%s %6i;  Value %4.6e' % (S, i, f0))
            s = np.subtract(np.multiply(np.divide(np.subtract(np.matmul(df3.conj().transpose(), df3), np.matmul(df0.conj().transpose(), df3)), np.matmul(df0.conj().transpose(), df0)), s), df3)    # Polack-Ribiere CG direction
            df0 = df3                                                       # swap derivatives
            d3 = d0
            d0 = np.matmul(df0.conj().transpose(), s)
            if d0 > 0:                                                      # new slope must be negative
                s = -df0
                d0 = np.matmul(-s.conj().transpose(), s)                    # otherwise use steepest direction
            x3 = np.multiply(x3, min(RATIO, np.divide(d3, (np.subtract(d0, np.finfo(np.double).tiny)))))                       # slope ratio but max RATIO
            ls_failed = 0                                                   # this line search did not fail
        else:
            # Restore best point so far
            loghyper = X0
            f0 = F0
            df0 = dF0
            if ls_failed or i > abs(length):                                # line search failed twice in a row
                break                                                     # or we ran out of time, so we give up
            s = -df0
            d0 = np.matmul(-s.conj().transpose(), s)                        # try steepest
            x3 = np.divide(1, (np.subtract(1, d0)))                     
            ls_failed = 1                                                   # this line search failed

    return [loghyper, fX]
