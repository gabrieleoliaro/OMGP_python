# OMGP_python
This is the code for a 2D Gaussian Process Regression based upon the paper "Overlapping Mixtures of Gaussian Processes for the data association problem" by Miguel Lázaro-Gredilla et al.

The code is not very fast, so it needs to be optimized.
Something to do is to make the program available on python versions earlier than 3.x, so that it can be run with the simple 'python' command
It would also help to rewrite/refactor the code into OOP

**Files:**<br/>
* `covariance.py` -- . <br/>
* `minimize.py` -- .  <br/>
* `omgp.py` -- .  <br/>
* `omgp_load.py` -- . <br/>
* `omgpbound.py` -- . <br/>
* `omgpEinc.py` -- .  <br/>
* `parser.py` -- .  <br/>
* `quality.py` -- . <br/>
* `sq_dist.py` -- . <br/>
* `test_omgp.py` -- The main module used to run the program.  <br/>
* `test.py` -- Module to read and print matrixes from and to xlsx files through pandas.  <br/>

Dependencies
============
* Matplotlib
* Numpy
* CSV
* (Pandas) - only for testing
