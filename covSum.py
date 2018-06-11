#!/usr/bin/python

def covSum(covfunc, loftheta, x, z):
    """
        covSum - compose a covariance function as the sum of other covariance
        functions. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work.

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.
    """

    for f in covfunc:                           # Iterate over covariance functions
        # TODO: translate if iscell(f{:}), f = f{:}; end          # Dereference cell array if necessary
        # TODO: translate j(i) = cellstr(feval(f{:}))

    narguments = len(args)
    if narguments == 1:                                   # Report number of parameters
        A = char(j(1))
        for i in range(2, len(covfunc)):
            # TODO: translate A = [A, '+', char(j(i))]
        return

    [n, D] = np.shape(x)

    v = []                          # v vector indicates to which covariance parameters belong
    for i in range(len(covfunc)):
        v = [v np.tile(i, (1, eval(char(j(i)))))]
        

    if narguments == 3:                     # Compute covariance matrix
        A = np.zeros(n, n);                   # Allocate space for covariance matrix
        for i in range(len(covfunc)):         # Iteration over summand functions
            f = covfunc(i);
            # TODO: translate if iscell(f{:}), f = f{:}; end        # Dereference cell array if necessary
                A = A + f{:}(logtheta(v==i), x));                   # Accumulate covariances

    if narguments == 4:                     # Compute derivative matrix or test set covariances
        #TODO translate if nargout == 2:    # Compute test set cavariances
            A = zeros((z.shape[0], 1))
            B = zeros((x.shape[0], z.shape[0]));    # Allocate space
            for i in len(covfunc):
                f = covfunc[i]
                #TODO translate if iscell(f{:}), f = f{:}; end      # dereference cell array if necessary
                [AA BB] = f{:}(logtheta(v==i), x, z)        # Compute test covariances
                A = A + AA
                B = B + BB                                  # And accumulate
        else:                                               # Compute derivative matrices
            i = v[z]                                       # which covariance function
            j = (v(1:z)==i).sum(axis=0)                    # which parameter in that covariance
            f = covfunc[i]
            # TODO translate: if iscell(f{:}), f = f{:}; end        # Dereference cell array if necessary
            A = f{:}(logtheta(v==i), x, j)                # Compute derivative
  

