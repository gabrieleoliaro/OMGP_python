#!/usr/bin/python

import sys
import numpy as np
from omgpbound import *

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


    i = 0                                           # Run length counter, initialized to zero
    ls_failed = 0                                  # No previous line search has failed
    [f0, df0] = f(loghyper, learn, covfunc, M, X, Y)                          # Get function value and gradient
    loghyper =2
    conv2=3
    return [loghyper, conv2]
