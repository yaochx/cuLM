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

#if DOUBLE_PRECISION == 1
    #define LM_MACHEP     DBL_EPSILON   /* resolution of arithmetic */
    #define LM_DWARF      DBL_MIN       /* smallest nonzero number */
    #define LM_SQRT_DWARF sqrt(DBL_MIN) /* square should not underflow */
    #define LM_SQRT_GIANT sqrt(DBL_MAX) /* square should not overflow */
    #define FLOAT         double
    #define BLOCK_SIZE    1248
#else
    #define LM_MACHEP     FLT_EPSILON   /* resolution of arithmetic */
    #define LM_DWARF      FLT_MIN       /* smallest nonzero number */
    #define LM_SQRT_DWARF sqrt(FLT_MIN) /* square should not underflow */
    #define LM_SQRT_GIANT sqrt(FLT_MAX) /* square should not overflow */
    #define FLOAT         float
    #define BLOCK_SIZE    1245
#endif

/*
    const int BLOCK_SIZE = 2*m  // X
                         + m    // Y
                         + n    // A
                         + m    // fvec
                         + n    // diag
                         + n    // qtf
                         + n*m  // fjac
                         + n    // wa1
                         + n    // wa2
                         + n    // wa3
                         + m    // wa4
                         + n;   // ipvt -> this is integer! but it is the same size as float anyway (4B)
    
    BLOCK_SIZE = 1245 items
    */

#define DATAX_SIZE 2*m
#define DATAY_SIZE m
#define DATAA_SIZE n
#define FVEC_SIZE m
#define DIAG_SIZE n
#define QTF_SIZE n
#define FJAC_SIZE n*m
#define WA1_SIZE n
#define WA2_SIZE n
#define WA3_SIZE n
#define WA4_SIZE m
#define IPVT_SIZE n

/* If the above values do not work, the following seem good for an x86:
 LM_MACHEP     .555e-16
 LM_DWARF      9.9e-324	
 LM_SQRT_DWARF 1.e-160   
 LM_SQRT_GIANT 1.e150 
 LM_USER_TOL   1.e-14
   The following values should work on any machine:
 LM_MACHEP     1.2e-16
 LM_DWARF      1.0e-38
 LM_SQRT_DWARF 3.834e-20
 LM_SQRT_GIANT 1.304e19
 LM_USER_TOL   1.e-14
*/

#ifdef __cplusplus
extern "C" {
#endif

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


#ifdef __cplusplus
}
#endif

#endif /* LMMIN_H */
