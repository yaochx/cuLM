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
    
    if(cuModuleLoad(&cuModule, module) != CUDA_SUCCESS)
        exit(1);
    
    if(cuModuleGetFunction(&cuLMmin, cuModule, function) != CUDA_SUCCESS)
        exit(1);
    
    if(cuLaunchKernel(cuLMmin, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL) != CUDA_SUCCESS)
        exit(1);
}

/* machine-dependent constants from FLOAT.h */
#define LM_USERTOL 30*DBL_EPSILON  // works also for single precision
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

FLOAT ** lm_minimize(int n_molecules, int m_dat, int n_par, FLOAT *par, lm_data_type *data, lm_control_type *control)
{
    const int NMOLECULES = n_molecules;
    const int FSIZE = sizeof(FLOAT);

    initCUDA();

    int n = n_par;
    int m = m_dat;
    
    CUdeviceptr memory;
    if((cuMemAlloc(&memory, BLOCK_SIZE*FSIZE*NMOLECULES) == cudaErrorMemoryAllocation))
    {
	    control->info = 9;
	    exit(9);
    }

    // Calculate the base addresses of each vector/matrix.
    CUdeviceptr X, Y, A, fvec, diag, fjac, qtf, wa1, wa2, wa3, wa4, ipvt;
    X    = memory;
    Y    = X    + (DATAX_SIZE*NMOLECULES*FSIZE);
    A    = Y    + (DATAY_SIZE*NMOLECULES*FSIZE);
    fvec = A    + (DATAA_SIZE*NMOLECULES*FSIZE);
    diag = fvec + ( FVEC_SIZE*NMOLECULES*FSIZE);
    fjac = diag + ( DIAG_SIZE*NMOLECULES*FSIZE);
    qtf  = fjac + ( FJAC_SIZE*NMOLECULES*FSIZE);
    wa1  = qtf  + (  QTF_SIZE*NMOLECULES*FSIZE);
    wa2  = wa1  + (  WA1_SIZE*NMOLECULES*FSIZE);
    wa3  = wa2  + (  WA2_SIZE*NMOLECULES*FSIZE);
    wa4  = wa3  + (  WA3_SIZE*NMOLECULES*FSIZE);
    ipvt = wa4  + (  WA4_SIZE*NMOLECULES*FSIZE);

    // Initialize host array and copy it to CUDA device
    for(int i = 0; i < NMOLECULES; i++)
    {
        cuMemcpyHtoD(X+i*(DATAX_SIZE*FSIZE), data->X, DATAX_SIZE*FSIZE);
        cuMemcpyHtoD(Y+i*(DATAY_SIZE*FSIZE), data->y, DATAY_SIZE*FSIZE);
        cuMemcpyHtoD(A+i*(DATAA_SIZE*FSIZE), par    , DATAA_SIZE*FSIZE);
    }


    // Do calculation on device (this goes through the modified legacy interface)
    int maxcall = control->maxcall * (n + 1), one = 1;
    void *args[] = { &m, &n, &A, &fvec, &(control->ftol), &(control->xtol), &(control->gtol),
                     &maxcall, &(control->epsilon), &diag, &one, &(control->stepbound),
                     &fjac, &ipvt, &qtf, &wa1, &wa2, &wa3, &wa4, &X, &Y, (void *)&NMOLECULES };

    // working in 1D only ... there is only 48kB od shared memory per block, thus with arrays of length 1248, it is possible to run 4 threads with double precision (8B per item) or 8 threads with single precision (4B per item)
    int blockX = 32 / FSIZE;
    int nblocks = (NMOLECULES / blockX) + (int)(NMOLECULES % blockX > 0);    // --> ceil(NMOLECULES / blockX)

    // run
#ifdef _DEBUG
    runCUDA("Debug/lmmin.ptx", "lmmin", dim3(blockX,1,1), dim3(nblocks,1,1), args);
#else
    runCUDA("Release/lmmin.ptx", "lmmin", dim3(blockX,1,1), dim3(nblocks,1,1), args);
#endif

    // Retrieve result from device and store it in host array
    FLOAT **results = new FLOAT*[NMOLECULES];
    for(int i = 0; i < NMOLECULES; i++)
    {
        results[i] = new FLOAT[DATAA_SIZE];
        cuMemcpyDtoH(results[i], A+i*(DATAA_SIZE*FSIZE), DATAA_SIZE*FSIZE);
    }
    
	// Clean up
    cuMemFree(memory);
    cleanCUDA();

    return results;
}

// ------------------------------------------------------------------------
// Function to fit
// ------------------------------------------------------------------------
#define DIM 2
#define NPAR 5

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
FLOAT ** lmfit2DgaussXYSAO(int n_molecules, FLOAT *par0, int npts, FLOAT *X, FLOAT *y, int ny, Options *options)
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
	FLOAT ** res = lm_minimize(n_molecules, data.npts, data.npar, par, &data, &control);
	for(int i = 0; i < n_molecules; i++)
    {
        // get function parameters - use transformation again
	    res[i][2] = SQR(res[i][2]);
	    res[i][3] = SQR(res[i][3]);
	    res[i][4] = SQR(res[i][4]);
    }
    
    // residual error 
	//FLOAT reserr = sumreserr(par, data.npts, &data);

	// exit flag
	//int exitflag = control.info;

	// # iterations
	//int iter = control.nfev;

	// message
	//const char *message = lm_infmsg[control.info];

	return res;
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
	assert(fitregionsize % 2 == 1);	// TODO: replace with Java exception

	FLOAT *a0 = (FLOAT *)malloc(5*sizeof(FLOAT));
	a0[0] = 0; a0[1] = 0; a0[3] = 1.6f; a0[4] = min(im,fitregionsize2); a0[2] = max(im,fitregionsize2)-a0[4];	// initial parameters
	
	// <--[static]
	FLOAT *X = (FLOAT *)malloc(2*fitregionsize2*sizeof(FLOAT));
	for(int i = -boxsize, idx = 0; i <= +boxsize; i++)
		for(int j = -boxsize; j <= +boxsize; j++)
			{ X[idx++] = (FLOAT)i; X[idx++] = (FLOAT)j; }
	// [static]-->
	
    const int n_molecules = 500000;
	Options options;	// [static]
	FLOAT **a = lmfit2DgaussXYSAO(n_molecules, a0, fitregionsize2, X, im, fitregionsize2, &options);
	// TODO: I should be able to recieve chi2 if i want to
	// if exitflag != ok then throw new JavaException(message);

    // expected result (with double precision): 0.0017, 0.056, 356.73, 0.94857, 182.76
    for(int i = 0; i < n_molecules; i++)
	    printf("x=%f,y=%f,I=%f,s=%f,b=%f\n",a[i][0],a[i][1],a[i][2],a[i][3],a[i][4]);
    
	return 0;
}