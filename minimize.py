#!/usr/bin/python

import sys
import numpy as np
from omgpbound import *
from test import *

def minimize(loghyper, f, length, learn, covfunc, M, X, Y):
    """
    Minimize a differentiable multivariate function using conjugate gradients.
    

    Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
    X           initial guess; may be of any type, including struct and cell array
    f           the name or pointer to the function to be minimized. The function
                f must return two arguments, the value of the function, and it's
                partial derivatives wrt the elements of X. The partial derivative
                must have the same type as X.
    length      length of the run; if it is positive, it gives the maximum number of
                line searches, if negative its absolute gives the maximum allowed
                number of function evaluations. Optionally, length can have a second
                component, which will indicate the reduction in function value to be
                expected in the first line-search (defaults to 1.0).
    P1, P2, ... parameters are passed to the function f.

    X           the returned solution
    fX          vector of function values indicating progress made
    i           number of iterations (line searches or function evaluations,
                depending on the sign of "length") used at termination.

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

    See also: checkgrad

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

    print('%s %6i;  Value %4.6e\n' % (S, i, f0))
    fX = f0
    i = i + (length<0)                                                  # count epochs?!

    s = -df0
    d0 = np.matmul(-s.conj().transpose(), s)                            # initial search direction (steepest) and slope
    x3 = red / (1 - d0)                                                 # initial step is red/(|s|+1)

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
                        [f3, df3] = omgpboundA((loghyper + np.multiply(x3, s)), learn, covfunc, M, X, Y) # (X + x3 * s) is loghyper
                        
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise Warning('')
                    success = 1
                except:                                                     # catch any error which occured in f
                    x3 = (x2 + x3) / 2                                      # bisect and try again
    
            if f3 < F0:                                                     # keep best values
                X0 = loghyper + np.multiply(x3, s)
                F0 = f3
                dF0 = df3
            d3 = np.matmul(df3.conj().transpose(), s)                      # new slope
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or m == 0:          # are we done extrapolating?
                break 
      
            # Move point 2 to point 1
            x1 = x2
            f1 = f2
            d1 = d2

            # Move point 3 to point 2
            x2 = x3
            f2 = f3
            d2 = d3

##            print("d0: %f\n" %(d0))
##            print("d1: %f\n" %(d1))
##            print("d2: %f\n" %(d2))
##            print("d3: %f\n" %(d3))
##            print("EXT: %d\n" %(EXT))
##            print("f0: %f\n" %(f0))
##            print("F0: %f\n" %(F0))
##            print("f1: %f\n" %(f1))
##            print("f2: %f\n" %(f2))
##            print("f3: %f\n" %(f3))
##            print("fX: %f\n" %(fX))
##            print("INT: %f\n" %(INT))
##            print("length: %f\n" %(length))
##            print("ls_failed: %f\n" %(ls_failed))
##            print("m: %f\n" %(m))
##            print("MAX: %f\n" %(MAX))
##            print("RATIO: %f\n" %(RATIO))
##            print("red: %f\n" %(red))
##            print("RHO: %f\n" %(RHO))
##            print("SIG: %f\n" %(SIG))
##            print("x1: %f\n" %(x1))
##            print("x2: %f\n" %(x2))
##            print("x3: %f\n" %(x3))
##            print_matrix(df0, 'df0')
##            print_matrix(dF0, 'dF0f')
##            print_matrix(df3, 'df3')
##            print_matrix(s, 's')
##            print_matrix(loghyper, 'X')
##            print_matrix(X0, 'x0')
##            print_matrix(Z, 'Z')

            # Make cubic extrapolation
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * np.power((x2 - x1), 2) / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))    # num. error possible, ok!
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:            # num prob | wrong sign?
                x3 = x2 * EXT                                                       # extrapolate maximum amount
            elif x3 > x2 * EXT:                                             # new point beyond extrapolation limit?
                x3 = x2 * EXT                                               # extrapolate maximum amount
            elif x3 < x2 + INT * (x2 - x1):                                 # new point too close to previous point?
                x3 = x2 + INT * (x2 - x1)
#DEBUG FROM HERE
        while (abs(d3) > - SIG * d0 or f3 > f0 + x3 * RHO * d0) and m > 0:  # keep interpolating
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:                           # choose subinterval
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
                x3 = x2 - (0.5 * d2 * (x4 - x2) ^2) / (f4 - f2 - d2 * (x4 - x2))    # quadratic interpolation
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)                       # cubic interpolation
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2)^2) - B) / A           # num. error possible, ok!
    
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2                                          # if we had a numerical problem then bisect
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))   # don't accept too close
            #[f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
            if f3 < F0:
                # keep best values
                X0 = loghyper + np.multiply(x3, s)
                F0 = f3
                dF0 = df3 
            m = m - 1
            i = i + (length < 0)                                            # count epochs?!
            d3 = np.matmul(df3.conj().transpose(), s)                       # new slope
        # end interpolation
    

        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:                 # if line search succeeded
            X = loghyper + np.matmul(x3, s)
            f0 = f3
            fX = [fX.conj().transpose(), f0].conj().transpose()             # update variables
            print('%s %6i;  Value %4.6e\n' % (S, i, f0))
            s = (df3.conj().transpose() * df3 - df0.conj().transpose() * df3) / (df0.conj().transpose() * df0) * s - df3    # Polack-Ribiere CG direction
            df0 = df3                                                       # swap derivatives
            d3 = d0
            d0 = df0.conj().transpose() * s
            if d0 > 0:                                                      # new slope must be negative
                s = -df0
                d0 = -s.conj().transpose() * s                              # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3 / (d0 - realmin))                       # slope ratio but max RATIO
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
            x3 = 1 / (1 - d0)                     
            ls_failed = 1                                                   # this line search failed

    
    loghyper =2
    conv2=3





































    
    return [loghyper, conv2]
