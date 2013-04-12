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
#include "lmmin.cuh"

#include <cuda.h>
#include <builtin_types.h>
//#include <helper_cuda_drvapi.h>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

inline FLOAT SQR(FLOAT x)	 { return (x*x); }
inline int SQR(int x) { return (x*x); }

// ------------------------------------------------------------------------
// CUDA stuff
// ------------------------------------------------------------------------
CUcontext cuContext;

void initCUDA()
{
    if(cuInit(cudaDeviceScheduleAuto) != CUDA_SUCCESS)   // initialize CUDA driver which needs to be done, since this uses the PTX modules
        exit(1);

    int deviceCount;
    CUdevice cuDevice;
    cuDeviceGetCount(&deviceCount);
    cuDeviceGet(&cuDevice, 0);
    //char name[100];
    //cuDeviceGetName(name, 100, cuDevice);
    //  static = cuDeviceGetAttribute(attribute, device_attribute, device);
    cuCtxCreate(&cuContext, 0, cuDevice);
}

void cleanCUDA()
{
    if(cuCtxDestroy(cuContext) != CUDA_SUCCESS)
        return; // exit? uz asi nejaky vysledky mit budu...
}

void runCUDA(const char *module, const char *function, dim3 block, dim3 grid, void **args)
{
    CUmodule cuModule;
    CUfunction cuLMmin;
    
    int m = *((int *)args[0]), n = *((int *)args[1]);
    unsigned int is = sizeof(int), fs = sizeof(FLOAT);
    
    // what exactly is this??!
    unsigned int memsize = 0;//256+n*is+m*fs+2*m*fs+5*n*fs+m*n*fs;  // 256 for local variables

    if(cuModuleLoad(&cuModule, module) != CUDA_SUCCESS)
        exit(1);
    
    if(cuModuleGetFunction(&cuLMmin, cuModule, function) != CUDA_SUCCESS)
        exit(1);
    
    if(cuLaunchKernel(cuLMmin, grid.x, grid.y, grid.z, block.x, block.y, block.z, memsize, NULL, args, NULL) != CUDA_SUCCESS)
        exit(1);
}

/* machine-dependent constants from FLOAT.h */
#define LM_USERTOL 30*DBL_EPSILON  // workd also for single precision
        // if I use 30*FLT_EPSILON instead, the value is relatively large and the program does not work well
        // if needed, use 1.e-14, which works fine

void lm_initialize_control(lm_control_type *control)
{
    control->maxcall = 100;
    control->epsilon = LM_USERTOL;
    control->stepbound = 100.;
    control->ftol = LM_USERTOL;
    control->xtol = LM_USERTOL;
    control->gtol = LM_USERTOL;
}


/*** the following messages are indexed by the variable info. ***/

const char *lm_infmsg[] = {
    "fatal coding error (improper input parameters)",
    "success (the relative error in the sum of squares is at most tol)",
    "success (the relative error between x and the solution is at most tol)",
    "success (both errors are at most tol)",
    "trapped by degeneracy (fvec is orthogonal to the columns of the jacobian)"
    "timeout (number of calls to fcn has reached maxcall*(n+1))",
    "failure (ftol<tol: cannot reduce sum of squares any further)",
    "failure (xtol<tol: cannot improve approximate solution any further)",
    "failure (gtol<tol: cannot improve approximate solution any further)",
    "exception (not enough memory)",
    "exception (break requested within function evaluation)"
};

const char *lm_shortmsg[] = {
    "invalid input",
    "success (f)",
    "success (p)",
    "success (f,p)",
    "degenerate",
    "call limit",
    "failed (f)",
    "failed (p)",
    "failed (o)",
    "no memory",
    "user break"
};

void lm_minimize(int m_dat, int n_par, FLOAT *par, lm_data_type *data, lm_control_type *control)
{
    initCUDA();

    int n = n_par;
    int m = m_dat;
    int fs = sizeof(FLOAT);
    int is = sizeof(int);

    CUdeviceptr fvec, diag, fjac, qtf, wa1, wa2, wa3, wa4, ipvt;
    
    if((cuMemAlloc(&fvec,   m*fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&diag, n  *fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&qtf , n  *fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&fjac, n*m*fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&wa1 , n  *fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&wa2 , n  *fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&wa3 , n  *fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&wa4 ,   m*fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&ipvt, n  *is) == cudaErrorMemoryAllocation))
    {
	    control->info = 9;
	    exit(9);
    }

    // Initialize host array and copy it to CUDA device
    CUdeviceptr X, Y, A;
    if((cuMemAlloc(&X, 2*m*fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&Y,   m*fs) == cudaErrorMemoryAllocation) ||
       (cuMemAlloc(&A,   n*fs) == cudaErrorMemoryAllocation))
    {
        control->info = 9;
	    exit(9);
    }
    cuMemcpyHtoD(X, data->X, 2*m*fs);
    cuMemcpyHtoD(Y, data->y,   m*fs);
    cuMemcpyHtoD(A, par    ,   n*fs);


    // Perform the fit
    //control->info = 0;
    //control->nfev = 0;

    // Do calculation on device (this goes through the modified legacy interface)
    int maxcall = control->maxcall * (n + 1), one = 1;
    void *args[] = { &m, &n, &A, &fvec, &(control->ftol), &(control->xtol), &(control->gtol),
                     &maxcall, &(control->epsilon), &diag, &one, &(control->stepbound),
                     &fjac, &ipvt, &qtf, &wa1, &wa2, &wa3, &wa4, &X, &Y };
#ifdef _DEBUG
    runCUDA("Debug/lmmin.ptx", "lmmin", dim3(1,1,1), dim3(1,1,1), args);
#else
    runCUDA("Release/lmmin.ptx", "lmmin", dim3(1,1,1), dim3(1,1,1), args);
#endif

    // Retrieve result from device and store it in host array
    cuMemcpyDtoH(par, A, n*fs);

	//if ( control->info < 0 )
	//    control->info = 10;

    // Clean up
    cuMemFree(fvec);
    cuMemFree(diag);
    cuMemFree(qtf);
    cuMemFree(fjac);
    cuMemFree(wa1);
    cuMemFree(wa2);
    cuMemFree(wa3);
    cuMemFree(wa4);
    cuMemFree(ipvt);

    cleanCUDA();
}

// ------------------------------------------------------------------------
// Function to fit
// ------------------------------------------------------------------------
#define DIM 2
#define NPAR 5

// ------------------------------------------------------------------------
// Compute sum of residual errors
// ------------------------------------------------------------------------
/*FLOAT sumreserr(FLOAT *par, int npts, void *data)
{
	FLOAT *fvec, *tmp, sum = 0;
	int i;

	if ((fvec = (FLOAT *) malloc(npts*sizeof(FLOAT))) == NULL)
		printf("Not enough memory.\n");

	my_evaluate(par, npts, fvec, data, &i);

	for(i = 0, tmp = fvec; i < npts; i++)
		sum += SQR(*tmp++);
	
	free(fvec);
	
	return sum;
}*/

// ------------------------------------------------------------------------
// Control parameters
// ------------------------------------------------------------------------

struct Options
{
	int maxcall;
	FLOAT epsilon;
	FLOAT stepbound;
	FLOAT ftol;
	FLOAT xtol;
	FLOAT gtol;

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

FLOAT max(FLOAT *arr, int n)
{
	FLOAT res = arr[0];
	for(int i = 1; i < n; i++)
		if(arr[i] > res)
			res = arr[i];
	return res;
}

FLOAT min(FLOAT *arr, int n)
{
	FLOAT res = arr[0];
	for(int i = 1; i < n; i++)
		if(arr[i] < res)
			res = arr[i];
	return res;
}

// ------------------------------------------------------------------------
// MAIN 
// ------------------------------------------------------------------------
FLOAT * lmfit2DgaussXYSAO(FLOAT *par0, int npts, FLOAT *X, FLOAT *y, int ny, Options *options)
{ 
	lm_control_type		control;
    lm_data_type		data;
    FLOAT				*par;
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
	par = (FLOAT *)malloc(NPAR*sizeof(FLOAT));
	for(int i = 0; i < NPAR; i++)
		par[i] = par0[i];

	// initialize input data
    data.fun = NULL;
	data.dim = DIM;
		
	// inverse transform of initial function parameters
	data.npar = NPAR;
	par[2] = sqrt(par[2]);
	par[3] = sqrt(par[3]);
	par[4] = sqrt(par[4]);
	
	// get data variables   X
	data.npts = npts;
	data.X = X;
	
	// get measured function values     y ~ fun(X,par)
	data.y = y;

	// set fitting options
	lm_initialize_control(&control);
	init_control_userdef(&control, options);
	
    // do the fitting
	lm_minimize(data.npts, data.npar, par, &data, &control);

	// residual error 
	//FLOAT reserr = sumreserr(par, data.npts, &data);

	// exit flag
	//int exitflag = control.info;

	// # iterations
	//int iter = control.nfev;

	// message
	//const char *message = lm_infmsg[control.info];

	// get function parameters - use transformation again
	par[2] = SQR(par[2]);
	par[3] = SQR(par[3]);
	par[4] = SQR(par[4]);

	return par;
}

int main()
{
	// ----------- THIS IS WHAT I RECIEVE ---------------//
	const int fitregionsize = 11;
	const int boxsize = fitregionsize / 2;
	const int fitregionsize2 = SQR(fitregionsize);
	FLOAT im[] =	// fitregionsize2
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

	FLOAT *a0 = (FLOAT *)malloc(5*sizeof(FLOAT));
	a0[0] = 0; a0[1] = 0; a0[3] = 1.6f; a0[4] = min(im,fitregionsize2); a0[2] = max(im,fitregionsize2)-a0[4];	// initial parameters
	
	// <--[static]
	FLOAT *X = (FLOAT *)malloc(2*fitregionsize2*sizeof(FLOAT));
	for(int i = -boxsize, idx = 0; i <= +boxsize; i++)
		for(int j = -boxsize; j <= +boxsize; j++)
			{ X[idx++] = (FLOAT)i; X[idx++] = (FLOAT)j; }
	// [static]-->
	
	Options options;	// [static]
	FLOAT *a = lmfit2DgaussXYSAO(a0, fitregionsize2, X, im, fitregionsize2, &options);
	//[a,chi2,exitflag] = lmfit2DgaussXYSAO(a0,reshape(X(:),npts,2)', im(:)');
	// TODO: I should be able to recieve chi2 if i want to
	// if exitflag != ok then throw new JavaException(message);

	// expected results: 0.0017, 0.056, 356.73, 0.94857, 182.76
	printf("x=%f,y=%f,I=%f,s=%f,b=%f\n",a[0],a[1],a[2],a[3],a[4]);

	return 0;
}