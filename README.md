# OMGP_python
This is the code for a 2D Gaussian Process Regression based upon the paper "Overlapping Mixtures of Gaussian Processes for the data association problem" by Miguel Lázaro-Gredilla et al.

The code is not very fast, so it needs to be optimized.
Something to do is to make the program available on python versions earlier than 3.x, so that it can be run with the simple 'python' command
It would also help to rewrite/refactor the code into OOP

**Files:**<br/>
* `covariance.py` -- Contains all the functions to generate the covariance matrices, test set covariances, and derivative matrices. <br/>
* `minimize.py` -- Allows you to minimize a differentiable multivariate function using conjugate gradients.  <br/>
* `omgp.py` -- The main function setting up the initial hyperparameters and calling omgpEinc and omgp_bound in turns to solve the problem and make probabilistic predictions.  <br/>
* `omgp_load.py` -- Used to generate random input or parse it from file. <br/>
* `omgpbound.py` -- Allows you to computes the negative of the Marginalized Variational Bound (F) and its derivatives wrt loghyper (dF). <br/>
* `omgpEinc.py` -- Performs the E-step.  <br/>
* `parser.py` -- Draft of CSV parser for ROS log file coming from @ewerlopes player_tracker code. Eventually it will probably be included into the omgp_load file.  <br/>
* `quality.py` -- Computes NMSE and NLPD measures for test data. <br/>
* `sq_dist.py` -- a function to compute a matrix of all pairwise squared distances
    between two sets of vectors, stored in the columns of the two matrices, a
    (of size D by n) and b (of size D by m). If only a single argument is given
    or the second matrix is empty, the missing matrix is taken to be identical
    to the first.. <br/>
* `test_omgp.py` -- The main module used to run the program.  <br/>
* `test.py` -- Module to read and print matrixes from and to xlsx files through pandas.  <br/>

Dependencies
============
* Matplotlib
* Numpy
* CSV
* (Pandas) - only for testing
