/*
 * Project:  LevenbergMarquardtLeastSquaresFitting
 *
 * File:     lmmin.c
 *
 * Contents: Public interface to the Levenberg-Marquardt core implementation.
 *
 * Author:   Joachim Wuttke 2004-8 
 * 
 * Homepage: www.messen-und-deuten.de/lmfit
 *
 * Licence:  Public domain.
 */
 
#ifndef LMMIN_H
#define LMMIN_H

// precision: float/FLOAT ...float is usually at least 8x faster than FLOAT, because number of FLOAT processing units is restricted on the chip!
#define DOUBLE_PRECISION 0

#if DOUBLE_PRECISION == 0
    #define FLOAT float
#else
    #define FLOAT double
#endif

#ifdef __cplusplus
extern "C" {
#endif


/** Default data type for passing y(t) data to lm_evaluate **/

typedef struct {
    FLOAT *tvec;
    FLOAT *yvec;
    FLOAT (*f) (FLOAT t, FLOAT *par);
} lm_data_type_default;


/** Compact high-level interface. **/

/* Collection of control parameters. */
typedef struct {
    FLOAT ftol;      /* relative error desired in the sum of squares. */
    FLOAT xtol;      /* relative error between last two approximations. */
    FLOAT gtol;      /* orthogonality desired between fvec and its derivs. */
    FLOAT epsilon;   /* step used to calculate the jacobian. */
    FLOAT stepbound; /* initial bound to steps in the outer loop. */
    FLOAT fnorm;     /* norm of the residue vector fvec. */
    int maxcall;      /* maximum number of iterations. */
    int nfev;	      /* actual number of iterations. */
    int info;	      /* status of minimization. */
} lm_control_type;

/* Collection of data parameters. */
typedef struct {
	size_t npar;		// number of function parameters to be optimized
	size_t npts;		// number of data points
	size_t dim;		    // data dimension
	FLOAT *X;          // matrix of data variables; in Matlab stored as [dim x npts] 
	FLOAT *y;          // vector of measured function values     y ~ fun(X,par)
	FLOAT (*fun) (FLOAT *X, FLOAT *par); // fnc definition     y = fun(X,par)
} lm_data_type;

/* Initialize control parameters with default values. */
void lm_initialize_control(lm_control_type * control);

/* The actual minimization. */
void lm_minimize(int m_dat, int n_par, FLOAT *par, lm_data_type *data, lm_control_type *control);

extern const char *lm_infmsg[];
extern const char *lm_shortmsg[];

#ifdef __cplusplus
}
#endif

#endif /* LMMIN_H */
