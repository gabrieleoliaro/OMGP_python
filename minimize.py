#!/usr/bin/python

import sys
import numpy as np

def minimize(X, f, length, *args): #*args is the equivalent of Matlab's varargin?

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

    # The code assumes that the passed argument length is a list/array.
    length_int = length[1] # Save the integer value of the length in a local variable for simplicity
    
    # Extrapolate whether the length gives us the maximum number of line searches
    # or the mazimum allowed number of function evaluations
    if length_int > 0:
        S = 'Linesearch'
    else
        S = 'Function evaluation'

    
    # Check if length has the optional component
    if len(length) == 2:
        red = length[2]
    else:
        red = 1;


    i = 0                                           # Run length counter, initialized to zero
    ls_failed = 0;                                  # No previous line search has failed
    [f0 df0] = f(X, args)                           # Get function value and gradient
    Z = X
    X = unwrap(X)
    df0 = np.matrix(unwrap(df0))
    print("%s %6d;  Value %4.6e\n" % (S, i, f0));   # Print to screen

    sys.stdout.flush()                              # Equivalent of Matlab's 'if exist('fflush','builtin')
                                                    # fflush(stdout); end'??
    fX = np.matrix(f0)
    i = i + (length < 0)                            # Count epochs?!
    
    # Initial search direction (steepest) and slope
    s = - np.matrix(df0)                            # Make s a numpy matrix object to allow the use of the getH() function
    d0 = - s.getH() * s                             # where getH() translates Matlab's complex conjugate transpose operator '

    x3 = red / (1 - d0)                             # Initial step is red / ( |s| + 1)

    while (i < abs(length)):                        # While not finished
        i = i + (length > 0)                        # Count the iterations?!
        
        # Make a copy of current values
        X0 = X
        F0 = f0
        dF0 = df0

        if (length > 0):
            M = MAX
        else:
            M = min(MAX, -length - i)

        while 1:                                    # Keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0

            while not success and (M > 0):
                try:
                    M = M - 1
                    i = i + (length < 0);                         # Count epochs?!

                    [f3 df3] = f(rewrap(Z, X + x3 * s), args)
                    df3 = unwrap(df3)
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)): # TODO
                        raise Exception('')
                    success = 1
            
                # Catch any error which occured in f
                except:
                    x3 = (x2 + x3) / 2                                  # Bisect and try again

            if (f3 < F0):
                # Keep the best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
                # TODO again: d3 = df3'*s                                       # New slope
                if (d3 > SIG * d0) or (f3 > f0 + x3 * RHO * d0) or (M == 0):    # Are we done extrapolating?
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
                A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
                B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
                
                # Num. error possible, ok!
                x3 = x1 - d1 * (x2 - x1) ^ 2 / (B + sqrt(B * B - A * d1 * (x2 - x1)));
                
                # num prob | wrong sign?
                if not isreal(x3) or isnan(x3) or isinf(x3) or x3 < 0:
                    x3 = x2 * EXT                                 # Extrapolate maximum amount
                elif (x3 > x2 * EXT):                            # new point beyond extrapolation limit?
                    x3 = x2 * EXT                                   # extrapolate maximum amount
                elif (x3 < x2 + INT * (x2 - x1)):         # new point too close to previous point?
                    x3 = x2 + INT * (x2 - x1)
        # Extrapolation is over

        while (abs(d3) > - SIG * d0 or f3 > f0 + x3 * RHO * d0) and (M > 0):  # keep interpolating
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:                        # choose subinterval
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
                x3 = x2 - (0.5 * d2 * (x4 - x2) ^ 2) / (f4 - f2 - d2 * (x4 - x2))  # quadratic interpolation
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)                    # cubic interpolation
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (sqrt(B * B - A * d2 * (x4 - x2) ^2 ) - B) / A        # num. error possible, ok!
            if isnan(x3) or isinf(x3):
                x3 = (x2 + x4) / 2                                # if we had a numerical problem then bisect

            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))  # don't accept too close
            [f3 df3] = f(rewrap(Z, X + x3 * s), args)
            df3 = np.matrix(unwrap(df3))                        # Make df3 a numpy matrix object to allow the getH() command
            if f3 < F0:
                # Keep best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            M = M - 1
            i = i + (length < 0)                             # Count epochs?!
            d3 = df3.getH() * s;                                # New slope
        # Interpolation is over

        if abs(d3) < - SIG * d0 and f3 < f0 + x3 * RHO * d0:          # If line search succeeded
            X = X + x3 * s
            f0 = f3
            f0 = np.matrix(f0)                              # Make f0 a numpy matrix object
            fX = np.matrix([fX.getH() f0].getH())                    # Update variables
            print("%s %6d;  Value %4.6e\n" % (S, i, f0))
            sys.stdout.flush()                              # Equivalent of Matlab's 'if exist('fflush','builtin'), fflush(stdout); end'??
            s = (df3.getH() * df3 - df0.getH() * df3) / (df0.getH() * df0) * s - df3   # Polack-Ribiere CG direction
            df0 = df3                                       # Swap derivatives
            d3 = d0
            d0 = df0.getH() * s
            if (d0 > 0):                                      # New slope must be negative
                s = -df0
                d0 = - s.getH() * s                  # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3 / (d0 - realmin));          # slope ratio but max RATIO
            ls_failed = 0                              # this line search did not fail
        else:
            X = X0
            f0 = F0
            df0 = dF0                    # restore best point so far
            if ls_failed or i > abs(length):         # line search failed twice in a row
                break                               # or we ran out of time, so we give up
            s = -df0
            d0 = - s.getH() * s                                        # try steepest
            x3 = 1 / (1 -  d0)
            ls_failed = 1                                    # this line search failed

    X = rewrap(Z, X)
    print("\n")
    sys.stdout.flush()                              # Equivalent of Matlab's 'if exist('fflush','builtin'), fflush(stdout); end'??

    return [X, fX, i]

# TODO: complete the unwrap/rewrap functions to be able to work correctly with struct/cell arrays
def unwrap(s):
    """
        Extract the numerical values from "s" into the column vector "v". The
        variable "s" can be of any type, including struct and cell array.
        Non-numerical elements are ignored. See also the reverse rewrap.m.
    """

    v = []
    if s.isnumeric():
        v = s[:]                        # Numeric values are recast to column vector
    # TODO: elif s.isstruct()
        #v = unwrap(struct2cell(orderfields(s))) # alphabetize, conv to cell, recurse
    # elif s.iscell()
#        for i in range(len(s)):             # cell array elements are handled sequentially
#            v = [v unwrap(s{i})]
    return v

def rewrap(s, v):

if isnumeric(s)
    if numel(v) < numel(s):
        raise Exception("The vector for conversion contains too few elements")
    s = reshape(v(1:numel(s)), size(s))             # numeric values are reshaped
    v = v(numel(s)+1:end);                       # remaining arguments passed on
#elif isstruct(s)
#    [s p] = orderfields(s)
#    p(p) = 1:numel(p);                          # alphabetize, store ordering
#    [t v] = rewrap(struct2cell(s), v)                 # convert to cell, recurse
#    s = orderfields(cell2struct(t,fieldnames(s),1),p);  # conv to struct, reorder
#elif iscell(s)
#    for i = 1:numel(s)                          # cell array elements are handled sequentially
#    [s{i} v] = rewrap(s{i}, v);                 # other types are not processed

def rewrap(s,v):
    """
        Map the numerical elements in the vector "v" onto the variables "s" which can
        be of any type. The number of numerical elements must match; on exit "v"
        should be empty. Non-numerical entries are just copied.
        function [s v] = rewrap(s, v)
        
        if isnumeric(s)
        if numel(v) < numel(s)
        error('The vector for conversion contains too few elements')
        end
        s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
        v = v(numel(s)+1:end);                        % remaining arguments passed on
        elseif isstruct(s)
        [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
        [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
        s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
        elseif iscell(s)
        for i = 1:numel(s)             % cell array elements are handled sequentially
        [s{i} v] = rewrap(s{i}, v);
        end
        end                                             % other types are not processed

    """

    


if __name__ == '__main__':
    minimize()
