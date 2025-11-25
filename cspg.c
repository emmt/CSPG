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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include "cspg.h"

#define GAMMA 1.0e-04
#define LMAX  1.0e+30
#define LMIN  1.0e-30
#define SIGMA1  0.1
#define SIGMA2  0.9

static inline double min(double a, double b)
{
    return a < b ? a : b; // TODO: NaN?
}

static inline double max(double a, double b)
{
    return a > b ? a : b; // TODO: NaN?
}

#if 0
static inline double clamp(double x, double lo, double hi)
{
    return min(max(lo, x), hi);
}
#endif

/* Copy `src` into `dst`. */
static void copy(long n, double dst[], const double src[])
{
    for (long i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* Compute line-search trial: dst[i] = x[i] + alpha*d[i] (for all i). */
static void step(long n, double dst[], const double x[], double alpha, const double d[])
{
    for (long i = 0; i < n; ++i) {
        dst[i] = x[i] + alpha*d[i];
    }
}

/* Return the maximal value in vector `x` of length `n`. */
static double maximum(long n, const double x[])
{
    double xmax = x[0];
    for (long i = 1; i < n; ++i) {
        xmax = max(xmax, x[i]);
    }
    return xmax;
}

/* Compute the scalar product ⟨x,y⟩ of the vectors `x` and `y` if length `n`. */
static double inner(long n, const double x[], const double y[])
{
    double s = 0.0;
    for (long i = 0; i < n; ++i) {
        s += x[i]*y[i];
    }
    return s;
}

/* Perform the non-monotone line search. */
static void linesearch(cspg_context* ctx);

/* Call objective function callback. Return non-zero on error. */
static int compute_objective(cspg_context* ctx, const double x[], double* f)
{
    ctx->inform = 0;
    ctx->func(ctx->n, x, f, &ctx->inform, ctx->func_data);
    if (ctx->inform != 0) {
       ctx->status = -90;
       return -1;
   }
   ++ctx->fcnt;
   return 0;
}

/* Call gradient callback. Return non-zero on error. */
static int compute_gradient(cspg_context* ctx, const double x[], double g[])
{
    ctx->inform = 0;
    ctx->grad(ctx->n, x, g, &ctx->inform, ctx->grad_data);
    if (ctx->inform != 0) {
       ctx->status = -91;
       return -1;
   }
   ++ctx->gcnt;
   return 0;
}

/* Call projection callback. Do nothing if callback is `NULL`. Return non-zero on error. */
static int compute_projection(cspg_context* ctx, double x[])
{
    ctx->inform = 0;
    if (ctx->proj != NULL) {
        ctx->proj(ctx->n, x, &ctx->inform, ctx->proj_data);
        if (ctx->inform != 0) {
            ctx->status = -92;
            return -1;
        }
        ++ctx->pcnt;
    }
   return 0;
}

/* Compute the continuous-project-gradient and its sup-norm. */
static int project_gradient(cspg_context* ctx)
{
    long n = ctx->n;
    const double* x = ctx->x;
    const double* g = ctx->g;
    double* gp = ctx->gp;
    for (long i = 0; i < n; ++i) {
        gp[i] = x[i] - g[i];
    }
    if (compute_projection(ctx, gp) != 0) {
        return -1;
    }
    double nrm = 0.0;
    for (long i = 0; i < n; ++i) {
        gp[i] -= x[i];
        nrm = max(nrm, fabs(gp[i]));
    }
    ctx->gpsupn = nrm;
    return 0;
}

/* Simple observer that can be used to print iterates. */
static void print_iter(cspg_context* ctx, void* data)
{
    FILE* out = data;
    if (ctx->iter % 10 == 0) {
        fputs("\n ITER\t F\t\t GPSUPNORM\n", out);
    }
    fprintf(out, " %ld\t %e\t %e\n", ctx->iter, ctx->f, ctx->gpsupn);
}

cspg_context* cspg_context_create(long m, long n)
{
    if (m < 1 || n < 1) {
        errno = EINVAL;
        return NULL;
    }
    size_t size = sizeof(cspg_context);
    cspg_context* ctx = malloc(size);
    if (ctx == NULL) {
        return NULL;
    }
    memset(ctx, 0, size);
    ctx->m = m;
    ctx->n = n;
    ctx->lastfv = (double*)malloc(m*sizeof(double));
    if (ctx->lastfv == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->d = (double*)malloc(n*sizeof(double));
    if (ctx->d == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->g = (double*)malloc(n*sizeof(double));
    if (ctx->g == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->gnew = (double*)malloc(n*sizeof(double));
    if (ctx->gnew == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->gp = (double*)malloc(n*sizeof(double));
    if (ctx->gp == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->xbest = (double*)malloc(n*sizeof(double));
    if (ctx->xbest == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    ctx->xnew = (double*)malloc(n*sizeof(double));
    if (ctx->xnew == NULL) {
        cspg_context_destroy(ctx);
        return NULL;
    }
    for (long i = 0; i < m; ++i) {
        ctx->lastfv[i] = -INFINITY;
    }
    ctx->lmin = LMIN;
    ctx->lmax = LMAX;
    ctx->gamma = GAMMA;
    ctx->sigma1 = SIGMA1;
    ctx->sigma2 = SIGMA2;
    return ctx;
}

void cspg_context_destroy(cspg_context* ctx)
{
    if (ctx == NULL) {
        return;
    }
    if (ctx->lastfv != NULL) {
        free(ctx->lastfv);
        ctx->lastfv =  NULL;
    }
    if (ctx->d != NULL) {
        free(ctx->d);
        ctx->d =  NULL;
    }
    if (ctx->g != NULL) {
        free(ctx->g);
        ctx->g =  NULL;
    }
    if (ctx->gnew != NULL) {
        free(ctx->gnew);
        ctx->gnew =  NULL;
    }
    if (ctx->gp != NULL) {
        free(ctx->gp);
        ctx->gp =  NULL;
    }
    if (ctx->xbest != NULL) {
        free(ctx->xbest);
        ctx->xbest =  NULL;
    }
    if (ctx->xnew != NULL) {
        free(ctx->xnew);
        ctx->xnew =  NULL;
    }
    free(ctx);
}

void cspg(cspg_objective *func,
          void* func_data,
          cspg_gradient *grad,
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
          int *inform)
{
   cspg_context* ctx = cspg_context_create(m, n);
   if (ctx == NULL) {
       return;
   }
   ctx->func = func; ctx->func_data = func_data;
   ctx->grad = grad; ctx->grad_data = grad_data;
   ctx->proj = proj; ctx->proj_data = proj_data;
   if (obsv != NULL) {
       ctx->obsv = obsv;
       ctx->obsv_data = obsv_data;
   } else if (verb >= 2) {
       ctx->obsv = print_iter;
       ctx->obsv_data = stdout;
   }
   ctx->maxit = maxit;
   ctx->maxfc = maxfc;
   ctx->verb = verb;
   ctx->x = x;
   ctx->epsopt = epsopt;
   cspg_solve(ctx);
   if (f != NULL) {
       *f = ctx->f;
   }
   if (gpsupn != NULL) {
       *gpsupn = ctx->gpsupn;
   }
   if (iter != NULL) {
       *iter = ctx->iter;
   }
   if (fcnt != NULL) {
       *fcnt = ctx->fcnt;
   }
   if (gcnt != NULL) {
       *gcnt = ctx->gcnt;
   }
   if (pcnt != NULL) {
       *pcnt = ctx->pcnt;
   }
   if (status != NULL) {
       *status = ctx->status;
   }
   if (status != NULL) {
       *status = ctx->status;
   }
   if (inform != NULL) {
       *inform = ctx->inform;
   }
   cspg_context_destroy(ctx);
}

void cspg_solve(cspg_context* ctx)
{
    long m = ctx->m;
    long n = ctx->n;

    ctx->status = 0;
    ctx->inform = 0;

    /* Print problem information */
    if (ctx->verb) {
        fputs("============================================================================\n"
              " This is the SPECTRAL PROJECTED GRADIENT (SPG) for convex-constrained       \n"
              " optimization. If you use this code, please, cite:                          \n\n"
              " E. G. Birgin, J. M. Martinez and M. Raydan, Nonmonotone spectral projected \n"
              " gradient methods on convex sets, SIAM Journal on Optimization 10, pp.      \n"
              " 1196-1211, 2000, and                                                       \n\n"
              " E. G. Birgin, J. M. Martinez and M. Raydan, Algorithm 813: SPG - software  \n"
              " for convex-constrained optimization, ACM Transactions on Mathematical      \n"
              " Software 27, pp. 340-349, 2001.                                            \n"
              "============================================================================\n\n",
              stdout);
        fputs(" Entry to SPG.\n", stdout);
    }

   /* Project initial guess */
   if (compute_projection(ctx, ctx->x) != 0) {
       goto done;
   }

   /* Compute function and gradient at the initial point */
   if (compute_objective(ctx, ctx->x, &ctx->f) != 0) {
       goto done;
   }
   if (compute_gradient(ctx, ctx->x, ctx->g) != 0) {
       goto done;
   }

   /* Store functional value for the non-monotone line search. */
   for (long i = 0; i < m; ++i) {
      ctx->lastfv[i] = ctx->f;
   }

   /* Compute continuous-project-gradient and its sup-norm */
   if (project_gradient(ctx) != 0) {
       goto done;
   }

   /* Initial steplength */
   if (ctx->gpsupn > 0.0) {
       ctx->lambda = min(ctx->lmax, max(ctx->lmin, 1.0/ctx->gpsupn));
   } else {
       ctx->lambda = 0.0; // TODO convergence?
   }

   /* Initiate best solution and functional value */
   ctx->fbest = ctx->f;
   copy(ctx->n, ctx->xbest, ctx->x);

   /* Print initial information */
   if (ctx->obsv != NULL) {
       ctx->obsv(ctx, ctx->obsv_data);
       if (ctx->status != 0) {
           goto done;
       }
   }

   /* ==================================================================
      Main loop
      ================================================================== */
   while (ctx->gpsupn > ctx->epsopt && ctx->iter < ctx->maxit && ctx->fcnt < ctx->maxfc) {

       /* Iteration */
       ++ctx->iter;

       /* Compute first trial for line-search and call non-monotone line-search method. */
       step(ctx->n, ctx->xnew, ctx->x, -ctx->lambda, ctx->g);
       if (compute_projection(ctx, ctx->xnew) != 0) {
           goto done;
       }
       linesearch(ctx);
       if (ctx->status != 0) goto done;

       /* Set new functional value and save it for the non-monotone line search */
       ctx->f = ctx->fnew;
       ctx->lastfv[(ctx->iter) % ctx->m] = ctx->f;

       /* Gradient at the new iterate */
       if (compute_gradient(ctx, ctx->xnew, ctx->gnew) != 0) {
           goto done;
       }

       /* Compute sts = ⟨s,s⟩, sty = ⟨s,y⟩ with s = xnew - x and y = gnew - g. Compute the
          continuous-projected-gradient and its sup-norm. */
       double sts = 0.0;
       double sty = 0.0;
       for (long i = 0; i < n; ++i) {
           double s_i  = ctx->xnew[i] - ctx->x[i];
           double y_i  = ctx->gnew[i] - ctx->g[i];
           sts  += s_i*s_i;
           sty  += s_i*y_i;
           ctx->x[i]  = ctx->xnew[i];
           ctx->g[i]  = ctx->gnew[i];
       }
       if (project_gradient(ctx) != 0) {
           goto done;
       }

       /* Spectral steplength */
       if (sty > 0.0) {
           ctx->lambda = max(ctx->lmin, min(sts/sty, ctx->lmax));
       } else {
           ctx->lambda = ctx->lmax;
       }

       /* Best solution and functional value */
       if (ctx->f < ctx->fbest) {
           ctx->fbest = ctx->f;
           copy(ctx->n, ctx->xbest, ctx->x);
       }

       /* Print iteration information */
       if (ctx->obsv != NULL) {
           ctx->obsv(ctx, ctx->obsv_data);
           if (ctx->status != 0) {
               goto done;
           }
       }
   }

   /* ==================================================================
      End of main loop
      ================================================================== */

 done:
   if (ctx->iter > 0) {
       /* Finish returning the best point TODO avoid copy if x il already best */
       ctx->f = ctx->fbest;
       copy(ctx->n, ctx->x, ctx->xbest);
   }
   /* Write statistics */
   if (ctx->verb) {
       printf("\n");
       printf(" Number of variables                : %ld\n", ctx->n);
       printf(" Number of iterations               : %ld\n", ctx->iter);
       printf(" Number of functional evaluations   : %ld\n", ctx->fcnt);
       printf(" Number of gradient evaluations     : %ld\n", ctx->gcnt);
       printf(" Number of projections              : %ld\n", ctx->pcnt);
       printf(" Objective function value           : %e\n",  ctx->fbest);
       printf(" Sup-norm of the projected gradient : %e\n", ctx->gpsupn);
   }

   /* Termination flag TODO set in loop */
   if (ctx->gpsupn <= ctx->epsopt ) {
       ctx->status = 0;
       if (ctx->verb) {
           printf("\n Flag of SPG: Solution was found.\n");
       }
   } else if (ctx->iter >= ctx->maxit) {
       ctx->status = 1;
       if (ctx->verb) {
           printf("\n Flag of SPG: Maximum of iterations reached.\n");
       }
   } else {
       ctx->status = 2;
       if (ctx->verb) {
           printf("\n Flag of SPG: Maximum of functional evaluations reached.\n");
       }
   }
}

/* Nonmonotone line search with safeguarded quadratic interpolation */
static void linesearch(cspg_context* ctx)
{
    /* Compute the search direction given the feasible variables `x` at the start of
       the line-search and first feasible variables to try in `xnew`. */
    long n = ctx->n;
    for (long i = 0; i < n; ++i) {
        ctx->d[i] = ctx->xnew[i] - ctx->x[i];
    }
    /* Compute the parameters of the Armijo's stopping criterion. */
    double gtd = inner(ctx->n, ctx->g, ctx->d);
    double fmax = maximum(ctx->m, ctx->lastfv);
    /* Adjust the step length until one of the stopping criteria hold. */
    ctx->alpha = 1.0;
    while (1) {
        /* Evaluate objective function at trial point. */
        if (compute_objective(ctx, ctx->xnew, &ctx->fnew) != 0) {
            break;
        }
        /* Check for stopping criteria. */
        if (ctx->fnew <= fmax + ctx->gamma*ctx->alpha*gtd) {
            ctx->status = 0;
            break;
        }
        if (ctx->fcnt >= ctx->maxfc) {
            ctx->status = 2;
            break;
        }
        /* Reduce the step length by a safeguarded quadratic interpolation. */
        if (ctx->alpha <= ctx->sigma1) {
            ctx->alpha /= 2.0;
        } else {
            double num = -gtd*ctx->alpha*ctx->alpha;
            double den = 2.0*(ctx->fnew - ctx->f - ctx->alpha*gtd);
            double atmp = num/den;
            if (atmp >= ctx->sigma1 && atmp <= ctx->sigma2*ctx->alpha) {
                ctx->alpha = atmp;
            } else {
                ctx->alpha /= 2.0;
            }
        }
        /* Next trial */
        step(ctx->n, ctx->xnew, ctx->x, ctx->alpha, ctx->d);
    }
}
