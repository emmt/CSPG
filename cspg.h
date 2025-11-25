/*
  * This file is part of CSPG, a C implementation of the "Spectral Projected Gradient"
  * method of E.G. Birgin, J.M. Martinez and M. Raydan.
  *
  * This code is released under the MIT License and is freely available at
  * https://github.com/emmt/CSPG/
  *
  * The original code by the authors of the method can be found at the TANGO Project web
  * page (www.ime.usp.br/~egbirgin/tango/).
  */

#ifndef CSPG_H_
#define CSPG_H_ 1

/**
 * Structure storing all parameters, work-spaces, and results of the SPG algorithm.
 *
 * This structure is allocated and instantiated with default parameters by
 * cspg_context_create(). It can be used to solve any number of optimization problems (of
 * the same size) with cspg_solve(). The allocated resources are eventually released with
 * cspg_context_destroy().
 */
typedef struct cspg_context_ cspg_context;

/**
 * Callback to compute the objective function.
 *
 * @param n        Number of variables.
 *
 * @param x        Vector of variables.
 *
 * @param f        Address to write `f(x)` the value of the objective function at `x`.
 *
 * @param inform   On entry, `*inform` is set to `0`. The callback may set `*inform` to a
 *                 non-zero code to indicate a failure.
 *
 * @param data     Anything needed by the callback.
 */
typedef void cspg_objective(long n, const double x[], double* f, int* inform, void* data);

/**
 * Callback to compute the gradient of the objective function.
 *
 * @param n        Number of variables.
 *
 * @param x        Vector of variables.
 *
 * @param g        Vector to store `∇f(x)` the gradient of the objective function at `x`.
 *
 * @param inform   On entry, `*inform` is set to `0`. The callback may set `*inform` to a
 *                 non-zero code to indicate a failure.
 *
 * @param data     Anything needed by the callback.
 */
typedef void cspg_gradient(long n, const double x[], double g[], int* inform, void* data);

/**
 * Callback to project the variables onto the feasible set.
 *
 * @param n        Number of variables.
 *
 * @param x        Vector of variables. On entry, the variables may be unfeasible. On return
 *                 `x` is overwritten with its projection onto the feasible set.
 *
 * @param inform   On entry, `*inform` is set to `0`. The callback may set `*inform` to a
 *                 non-zero code to indicate a failure.
 *
 * @param data     Anything needed by the callback.
 */
typedef void cspg_projection(long n, double x[], int* inform, void* data);

/**
 * User observer callback.
 *
 * If provided, the observer callback is called by the SPG algorithm at every iteration. It
 * may print information. It may also set `ctx->status` to a non-zero value to terminate the
 * algorithm.
 *
 * @param ctx      Context of the algorithm.
 *
 * @param data     Anything needed by the callback.
 */
typedef void cspg_observer(cspg_context* ctx, void* data);

struct cspg_context_ {
    long m;
    long n;
    cspg_objective* func;
    void* func_data;
    cspg_gradient* grad;
    void* grad_data;
    cspg_projection* proj;
    void* proj_data;
    cspg_observer* obsv;
    void* obsv_data;

    double* lastfv;
    double* x;
    double f;
    double* g;
    double* gp;
    double gpsupn;

    double* xbest;
    double fbest;

    double alpha;
    double lambda;
    double* xnew;
    double fnew;
    double* gnew;

    double* d;
    double* s;
    double* y;

    double lmin;
    double lmax;
    double gamma;
    double sigma1;
    double sigma2;

    long maxit;
    long maxfc;
    double epsopt;

    int verb;
    long iter;
    long fcnt;
    long gcnt;
    long pcnt;
    int status; // algorithm status
    int inform; // code returned by user defined function (func, grad, or proj)
};

/**
 * Create a re-usable context for the SPG algorithm.
 *
 * This function allocates and instantiates with default parameters a context that can be
 * used to solve any number of optimization problems (with same `m` and `n`) with
 * cspg_solve(). Before calling cspg_solve(), the callbacks implementing the objective
 * function, its gradient, and the projection onto the feasible set must be instantiated.
 * The stopping criteria parameters (`ctx->epsopt`, `ctx->maxit`, and `ctx->maxfc`)
 * must be chosen and other parameters may also be changed.
 *
 * The caller is responsible of calling cspg_context_destroy() to free allocated resources.
 *
 * @param m    Number of previous objective function values to memorize, 'm ≥ 1`.
 *
 * @param n    Number of variables, 'n ≥ 1`.
 *
 * @return A context to run the SPG algorithm or `NULL` in case of error (with global
 *         `errno` code set to `ENOMEN` if there is insufficient memory, or to `EINVAL` if
 *         `m` or `n` have an invalid value).
 */
extern cspg_context* cspg_context_create(long m, long n);

/**
 * Free a context created for the SPG algorithm.
 *
 * This function frees the context structure and all allocated work-spaces.
 *
 * @param ctx  Context created by cspg_context_create().
 */
extern void cspg_context_destroy(cspg_context* ctx);

/**
 * Solve a constrained optimization problem by the SPG algorithm.
 *
 * This function minimizes an objective function with convex constraints by the "Spectral
 * Projected Gradient" (SPG) method (Version 2: "Feasible continuous projected path")
 * described in:
 *
 * - E.G. Birgin, J.M. Martinez and M. Raydan, *"Nonmonotone spectral projected gradient
 *   methods for convex sets"*, SIAM Journal on Optimization **10**, pp. 1196-1211 (2000).
 *
 * The user must supply the callbacks to evaluate the objective function and its gradient
 * and to project an arbitrary point onto the feasible region.
 *
 * @param ctx  Context created by cspg_context_create().
 */
extern void cspg_solve(cspg_context* ctx);

/**
 * Solve a constrained optimization problem by the SPG algorithm.
 *
 * This function minimizes an objective function with convex constraints by the "Spectral
 * Projected Gradient" (SPG) method (Version 2: "Feasible continuous projected path")
 * described in:
 *
 * - E.G. Birgin, J.M. Martinez and M. Raydan, *"Nonmonotone spectral projected gradient
 *   methods for convex sets"*, SIAM Journal on Optimization **10**, pp. 1196-1211 (2000).
 *
 * The user must supply the callbacks to evaluate the objective function and its gradient
 * and to project an arbitrary point onto the feasible region.
 *
 * @param func           User defined callback to compute the objective function (see
 *                       cspg_objective()).
 *
 * @param func_data      Anything needed by `func`.
 *
 * @param grad           User defined callback to compute the gradient of the objective
 *                       function (see cspg_gradient()).
 *
 * @param grad_data      Anything needed by `grad`.
 *
 * @param proj           If non-`NULL`, user defined callback to compute the projection of
 *                       the variables onto the feasible set (see cspg_projection()).
 *
 * @param proj_data      Anything needed by `proj`.
 *
 * @param obsv           If non-`NULL`, user defined observer callback (see
 *                       cspg_observer()).
 *
 * @param obsv_data      Anything needed by `obsv`.
 *
 * @param m              Number of previous function values to memorize, `m ≥ 1`.
 *
 * @param n              Number of variables, `n ≥ 1`.
 *
 * @param x              Initial variables on input, solution on output.
 *
 * @param epsopt         Tolerance for the convergence criterion.
 *
 * @param maxit          Maximum number of iterations.
 *
 * @param maxfc          Maximum number of functional evaluations.
 *
 * @param verb           Controls output level (0 = no print).
 *
 * @param f              If non-`NULL`, address to store the value of the objective function
 *                       the solution.
 *
 * @param gpsupn         If non-`NULL`, address to store the sup-norm of the projected
 *                       gradient at the solution.
 *
 * @param iter           If non-`NULL`, address to store the number of iterations.
 *
 * @param fcnt           If non-`NULL`, address to store the number of functional
 *                       evaluations.
 *
 * @param gcnt           If non-`NULL`, address to store the number of gradient evaluations.
 *
 * @param pcnt           If non-`NULL`, address to store the number of projections.
 *
 * @param status         If non-`NULL`, address to store the reason for stopping: `0` if
 *                       convergence criterion holds, `1` if the maximum number of
 *                       iterations has been reached, `2` if the maximum number of
 *                       functional evaluations has been reached, `-90` if an error occurred
 *                       in `func`, `-91` if an error occurred in `grad`, and `-92` if an
 *                       error occurred in `proj`.
 *
 * @param inform         If non-`NULL`, address to store the error code reported by one of
 *                       the callbacks: `func`, `grad`, or `proj`.
 */
extern void cspg(cspg_objective* func,
                 void* func_data,
                 cspg_gradient* grad,
                 void* grad_data,
                 cspg_projection* proj,
                 void* proj_data,
                 cspg_observer* obsv,
                 void* obsv_data,
                 long m,
                 long n,
                 double x[],
                 double epsopt,
                 long maxit,
                 long maxfc,
                 int verb,
                 double *f,
                 double *gpsupn,
                 long *iter,
                 long *fcnt,
                 long *gcnt,
                 long *pcnt,
                 int *status,
                 int *inform);

#endif /* CSPG_H_ */
