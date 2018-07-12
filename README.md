# OMGP_python
This code aims to associate observations of moving entities around a robot to either players and/or objects in the surrounding space.
- The approach is to use Overlapping Mixtures of Gaussian Process Experts (OMGP), as in the 2011 [paper](https://arxiv.org/abs/1108.3372) by Lázaro-Gredilla et al.
- The robot is powered by the Robotics Operating System (ROS) and needs to be able to interact with the player and play a dynamic game. It uses lasers and a Microsoft Kinect ® camera to perform SLAM and to locate the player(s) 

The code is not too fast yet (because of the slowless of the np.linalg.solve function, see [cProfile results](//github.com/gabrieleoliaro/OMGP_python/snakeviz%20cProfile.pdf)), but it will be soon optimized.

**Files:**<br/>
* `covariance.py` -- Contains all the functions to generate the covariance matrices, test set covariances, and derivative matrices. <br/>
* `minimize.py` -- Allows you to minimize a differentiable multivariate function using conjugate gradients.  <br/>
* `omgp.py` -- The main function setting up the initial hyperparameters and calling omgpEinc and omgp_bound in turns to solve the problem and make probabilistic predictions.  <br/>
* `omgp_load.py` -- Used to generate random input or parse it from file. <br/>
* `omgpbound.py` -- Allows you to computes the negative of the Marginalized Variational Bound (F) and its derivatives wrt loghyper (dF). <br/>
* `omgpEinc.py` -- Performs the E-step.  <br/>
* `parser.py` -- Draft of CSV parser for ROS log file coming from [Ewerton Lopes](//github.com/ewerlopes)'s player_tracker code. Eventually it will probably be included into the omgp_load file.  <br/>
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
