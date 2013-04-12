// Matlab MEX
//
// Levenberg-Marquardt Least Squares Fitting of 2D Gaussian
//
// [par, exitflag, residua, numiter, message] = lmfit1DgaussXSAO(par0, X, y, options);
//
// Note that optimized function parameters have limited range such that sigma >= 0, amplitude >= 0, offset >= 0

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lmmin.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

typedef struct {
	size_t npar;		// number of function parameters to be optimized
	size_t npts;		// number of data points
	size_t dim;		    // data dimension
	double *X;          // matrix of data variables; in Matlab stored as [dim x npts] 
	double *y;          // vector of measured function values     y ~ fun(X,par)
	double (*fun) (double *X, double *par); // fnc definition     y = fun(X,par)
} my_data_type;

inline double SQR(double x)	 { return (x*x); }
inline int SQR(int x) { return (x*x); }

// ------------------------------------------------------------------------
// Function to fit
// ------------------------------------------------------------------------
#define DIM 2
#define NPAR 5
double my_fit_function(double *X, double *par)
{
	return par[2] * exp(-0.5 * ( SQR((X[0]-par[0])/par[3]) + SQR((X[1]-par[1])/par[3]) )  ) + par[4];
}

// Transformation of function parameters to limit their range
void partransf(double *parin, double *parout, size_t npar)
{
	parout[0] = parin[0];       // x center stays the same
	parout[1] = parin[1];       // y center stays the same
	parout[2] = SQR(parin[2]);  // amplitude >= 0
	parout[3] = SQR(parin[3]);  // sigma >= 0
	parout[4] = SQR(parin[4]);  // background >= 0
}

// Inverse transformation of function parameters
void parinvtransf(double *parin, double *parout, size_t npar)
{
	parout[0] = parin[0];
	parout[1] = parin[1];
	parout[2] = sqrt(parin[2]);
	parout[3] = sqrt(parin[3]);
	parout[4] = sqrt(parin[4]);
}

// ------------------------------------------------------------------------
// Compute residua for all data points
// ------------------------------------------------------------------------
void my_evaluate(double *par, int npts, double *fvec, void *data, int *info)
{
    my_data_type *mydata = (my_data_type *) data;
	double	*y = mydata->y, *X = mydata->X;
	int		i;
	double p[NPAR];
    
	// transform function parameters to limit their range
	partransf(par, p, mydata->npar);

	// compute the difference " F = y - fun(X, par) " for all data points
	for (i = 0; i < npts; i++, X += mydata->dim)
		*(fvec++) = *(y++) - mydata->fun(X, p);

	*info = *info;		// to prevent a 'unused variable' warning
}

// ------------------------------------------------------------------------
// Compute sum of residual errors
// ------------------------------------------------------------------------
double sumreserr(double *par, int npts, void *data)
{
	double *fvec, *tmp, sum = 0;
	int i;

	if ((fvec = (double *) malloc(npts*sizeof(double))) == NULL)
		printf("Not enough memory.\n");

	my_evaluate(par, npts, fvec, data, &i);

	for(i = 0, tmp = fvec; i < npts; i++)
		sum += SQR(*tmp++);
	
	free(fvec);
	
	return sum;
}

// ------------------------------------------------------------------------
// Control parameters
// ------------------------------------------------------------------------

struct Options
{
	int maxcall;
	double epsilon;
	double stepbound;
	double ftol;
	double xtol;
	double gtol;

	Options()
	{	// all deactivated as defaults
		maxcall = -1;
		epsilon = stepbound = ftol = xtol = gtol = -1.0;
	}
};

void init_control_userdef(lm_control_type * control, const Options *options)
{
	if(options->maxcall   >= 0  ) control->maxcall   = options->maxcall;
	if(options->epsilon   >= 0.0) control->epsilon   = options->epsilon;
	if(options->stepbound >= 0.0) control->stepbound = options->stepbound;
	if(options->ftol      >= 0.0) control->ftol      = options->ftol;
	if(options->xtol      >= 0.0) control->xtol      = options->xtol;
	if(options->gtol      >= 0.0) control->gtol      = options->gtol;
}

double max(double *arr, int n)
{
	double res = arr[0];
	for(int i = 1; i < n; i++)
		if(arr[i] > res)
			res = arr[i];
	return res;
}

double min(double *arr, int n)
{
	double res = arr[0];
	for(int i = 1; i < n; i++)
		if(arr[i] < res)
			res = arr[i];
	return res;
}

// ------------------------------------------------------------------------
// MAIN 
// ------------------------------------------------------------------------
double * lmfit2DgaussXYSAO(double *par0, int npts, double *X, double *y, int ny, Options *options)
{ 
	lm_control_type		control;
    my_data_type		data;
    double				*par;
	/*
	// test number of input arguments
	printf("\nLevenberg-Marquardt Least Squares Fitting of 1D Gaussian\n\n");
	printf(" Usage:\n");
	printf("  [par, reserr, exitflag, numiter, message] = lmfit1DgaussXSAO(par0, X, y, options);\n\n");
	printf(" In:\n");
	printf("   par0     ... [1 x npar]      initial function parameters [meanx, meany, sigma, amplitude, offset]\n");
	printf("   X        ... [dim x npts]    matrix of data variables\n");
	printf("   y        ... [1 x npts]      vector of measured function values y ~ fun(X,par) \n");
	printf("   options  ... struct with fileds: maxcall / epsilon / stepbound / ftol / xtol / gtol\n\n");
	printf(" Out:\n");
	printf("   par      ... [1 x npar]      optimized parameters [meanx, meany, sigma, amplitude, offset]\n");
	printf("   reserr   ...                 sum of residual errors: sum((y-fun(X,par)).^2)\n");
	printf("   exitflag ...                 status\n");
	printf("   numiter  ...                 number of iterations\n");
	printf("   message  ...                 string with status message\n\n");
	printf(" Note that following parameters have limited range: sigma >= 0, amplitude >= 0, offset >= 0\n\n");
	*/
	// initialize I/O parameters
	par = (double *)malloc(NPAR*sizeof(double));
	for(int i = 0; i < NPAR; i++)
		par[i] = par0[i];

	// initialize input data
    data.fun = my_fit_function;
	data.dim = DIM;
		
	// inverse transform of initial function parameters
	data.npar = NPAR;
	parinvtransf(par,par,data.npar);
	
	// get data variables   X
	data.npts = npts;
	data.X = X;
	
	// get measured function values     y ~ fun(X,par)
	data.y = y;

	// set fitting options
	lm_initialize_control(&control);
	init_control_userdef(&control, options);
	
    // do the fitting
	lm_minimize(data.npts, data.npar, par, my_evaluate, NULL, &data, &control);

	// residual error 
	double reserr = sumreserr(par, data.npts, &data);

	// exit flag
	int exitflag = control.info;

	// # iterations
	int iter = control.nfev;

	// message
	const char *message = lm_infmsg[control.info];

	// get function parameters - use transformation again
	partransf(par, par, data.npar);

	return par;
}

int main()
{
	// ----------- THIS IS WHAT I RECIEVE ---------------//
	const int fitregionsize = 11;
	const int boxsize = fitregionsize / 2;
	const int fitregionsize2 = SQR(fitregionsize);
	double im[] =	// fitregionsize2
	{
		176, 159, 177, 200, 157, 183, 179, 169, 185, 167, 182,
		176, 158, 173, 190, 202, 205, 174, 179, 178, 174, 167,
		174, 191, 173, 179, 181, 179, 190, 176, 196, 169, 172,
		190, 176, 172, 185, 202, 207, 201, 169, 193, 191, 167,
		198, 170, 190, 198, 285, 402, 309, 197, 166, 176, 163,
		214, 190, 178, 196, 388, 513, 423, 236, 199, 206, 170,
		204, 205, 193, 186, 311, 404, 273, 214, 178, 167, 149,
		199, 185, 191, 179, 199, 212, 216, 180, 180, 166, 171,
		201, 184, 200, 204, 204, 210, 174, 203, 166, 180, 171,
		180, 169, 184, 203, 199, 184, 187, 180, 178, 199, 194,
		201, 184, 188, 180, 198, 203, 198, 183, 194, 180, 196
	};


	// ---------- START ----------------//
	assert(fitregionsize % 2 == 1);	// TODO: replace by Java exception

	double *a0 = (double *)malloc(5*sizeof(double));
	a0[0] = 0; a0[1] = 0; a0[3] = 1.6; a0[4] = min(im,fitregionsize2); a0[2] = max(im,fitregionsize2)-a0[4];	// initial parameters
	
	// <--[static]
	double *X = (double *)malloc(2*fitregionsize2*sizeof(double));
	for(int i = -boxsize, idx = 0; i <= +boxsize; i++)
		for(int j = -boxsize; j <= +boxsize; j++)
			{ X[idx++] = i; X[idx++] = j; }
	// [static]-->
	
	Options options;	// [static]
	double *a = lmfit2DgaussXYSAO(a0, fitregionsize2, X, im, fitregionsize2, &options);
	//[a,chi2,exitflag] = lmfit2DgaussXYSAO(a0,reshape(X(:),npts,2)', im(:)');
	// TODO: I should be able to recieve chi2 if i want to
	// if exitflag != ok then throw new JavaException(message);

	// expected results: 0.0017, 0.056, 356.73, 0.94857, 182.76
	printf("x=%f,y=%f,I=%f,s=%f,b=%f\n",a[0],a[1],a[2],a[3],a[4]);

	return 0;
}