# C version of the Spectral Projected Gradient method

This repository implements a C version of the *"Spectral Projected Gradient"* (SPG) method
of E.G. Birgin, J.M. Martinez and M. Raydan.

The SPG method is described in:

* E. G. Birgin, J. M. Martinez and M. Raydan, *"Nonmonotone spectral projected gradient
  methods on convex sets"*, SIAM Journal on Optimization **10**, pp. 1196-1211, 2000.

* E. G. Birgin, J. M. Martinez and M. Raydan, *"Algorithm 813: SPG - software for
  convex-constrained optimization"*, ACM Transactions on Mathematical Software **27**, pp.
  340-349, 2001.

Improvements of this implementation:

* For more flexibility, the objective function, its gradient, and the projection onto the
  feasible set are provided by user-defined callbacks. Changing these does not require
  recompiling.

* An observer callback may be provided by the user to print information and/or stop the
  algorithm at any iteration.

* All work-spaces and parameters are stored into a *"context"* structure which may be
  re-used to solve other problems (of the same size) with the SPG algorithm or the same
  problem with different initial solution or parameters.

* The context is instantiated with default parameters reflecting the original algorithm but
  may be changed without recompiling.

* Possible rounding-errors in the first line-search trial are avoided which ensures that
  the variables are always feasible when the objective function and its gradient are
  computed.

* Minimal resources are allocated (e.g., no needs to allocate vectors `s` and `y`).
