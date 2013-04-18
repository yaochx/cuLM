/*
   REPORT
    - momentalni stav: nepada to a neco to dela...ovsem krome odhadu X jsou vsechno defaultni hodnoty
      --> muj tip je, ze je spatne CID_BASE, protoze x je prvni parametr (prvni adresa je zrejme spravne), takze to asi nebude nahoda

   TODO
    - funguje CID_BASE ve volani lmenorm tak, jak by mel?
    - zbavit se goto!
    - tykaji se me bank conflicts? nejak jsem nepobral, co to ma byt
    - zkusit presunout ty dva cykly pro Y a A, co jsou ve funkci LMfit(main.cpp) do inicializace v lmdiff, cimz by to bylo asi cistsi, zejmena pak z javy
    - vyber nejlepsiho zarizeni s nejvice GFlops
    - mereni single, nebo double?
    
    - vytvorit tagy u verzi, kde se merilo
    - pripadne vytvorit branch pro predchozi mereni bez pouziti shared pameti - uz ted totiz vim, ze jsem se nemusel tolik omezovat a v bloku mohlo byt vlaken hafo...
      --> najit nejlepsi pomer bloky/vlakna zvlast pro float a double a provest mereni
          --> pridat jeste navic coalescing - i u globalni pameti to ma vliv
          --> zkusit prepnout vic pameti do L1 cache, jak je napsano zde: https://developer.nvidia.com/content/using-shared-memory-cuda-cc, napr. cudaFuncCachePreferL1
              --> podobne bych tomhle mohl zkusit hejbat u shared verze...staci spocitat, kolik potrebuju bajtu na lokalni promenny a volani funkci a dat to k dispozici pres L1

    - Java-friendly interface v PTX
*/


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

// ------------------------------------------------------------------------
// CUDA stuff
// ------------------------------------------------------------------------
class CUDA
{
    private:
        CUcontext m_cuContext;

    protected:
        void init()
        {
            if(cuInit(cudaDeviceScheduleAuto) != CUDA_SUCCESS)
                throw("cuInit");

            int deviceCount;
            CUdevice cuDevice;
            cuDeviceGetCount(&deviceCount);
            cuDeviceGet(&cuDevice, 0);

            //char name[100];
            //cuDeviceGetName(name, 100, cuDevice);
            //  static = cuDeviceGetAttribute(attribute, device_attribute, device);

            cuCtxCreate(&m_cuContext, 0, cuDevice);
        }

        void clean()
        {
            if(cuCtxDestroy(m_cuContext) != CUDA_SUCCESS)
                throw("cuCtxDestroy");
        }

    public:
        CUDA()
        {
            init();
        }

        ~CUDA()
        {
            clean();
        }

        void run(const char *module, const char *function, dim3 block, dim3 grid, void **args)
        {
            CUmodule cuModule;
            CUfunction cuLMmin;
    
            if(cuModuleLoad(&cuModule, module) != CUDA_SUCCESS)
                throw("cuModuleLoad");
    
            if(cuModuleGetFunction(&cuLMmin, cuModule, function) != CUDA_SUCCESS)
                throw("cuModuleGetFunction");
    
            // Invokes the kernel f on a gridDimX x gridDimY x gridDimZ grid of blocks. Each block contains blockDimX x blockDimY x blockDimZ threads.
            if(cuLaunchKernel(cuLMmin, grid.x, grid.y, grid.z, block.x, block.y, block.z, 8*1248*FSIZE, NULL, args, NULL) != CUDA_SUCCESS)
                throw("cuLaunchKernel");
        }

        CUdeviceptr alloc(int bytes)
        {
            CUdeviceptr d_memory;
            if((cuMemAlloc(&d_memory, bytes) == cudaErrorMemoryAllocation))
                throw("cuMemAlloc");
            return d_memory;
        }

        void free(CUdeviceptr d_ptr)
        {
            if((cuMemFree(d_ptr) == cudaErrorMemoryAllocation))
                throw("cuMemFree");
        }
};

// ------------------------------------------------------------------------
// cuLM fitting of 2D symmetric Gaussian
// ------------------------------------------------------------------------
class Gaussian2DFitting
{
    private:
        int n_params;
        int n_molecules;
        int n_boxsize;
        int n_fitting_region;
        int n_input_data;
        
        int control_maxcall;
        FLOAT control_epsilon;
        FLOAT control_stepbound;
        FLOAT control_ftol;
        FLOAT control_xtol;
        FLOAT control_gtol;

        template<typename T> T sqr(T x)
        {
            return x*x;
        }

        void TransformForwardParams(FLOAT **params)
        { 
            for(int i = 0; i < n_molecules; i++)
            {
                params[i][2] = sqrt(params[i][2]);
	            params[i][3] = sqrt(params[i][3]);
	            params[i][4] = sqrt(params[i][4]);
            }
        }

        void TransformBackParams(FLOAT **params)
        { 
            for(int i = 0; i < n_molecules; i++)
            {
                params[i][2] = sqr(params[i][2]);
	            params[i][3] = sqr(params[i][3]);
	            params[i][4] = sqr(params[i][4]);
            }
        }

    public:
        Gaussian2DFitting(int boxsize)
        {
            n_boxsize = boxsize;
            n_fitting_region = 1+2*boxsize;
            n_input_data = sqr(n_fitting_region);
            n_params = 5;

            control_maxcall = 100;
            control_stepbound = 100.0;
            control_epsilon = 30*DBL_EPSILON;
            control_ftol = 30*DBL_EPSILON;
            control_xtol = 30*DBL_EPSILON;
            control_gtol = 30*DBL_EPSILON;

            // Note: 30*DBL_EPSILON works also with single precision
            // if I use 30*FLT_EPSILON instead, the value is relatively large and the program does not work well
            // if needed, use 1.e-14, which should work fine with any machine
        }

        void SetControlParameters(int maxcall, FLOAT stepbound, FLOAT epsilon, FLOAT ftol, FLOAT xtol, FLOAT gtol)
        {
            if(ftol >= 0.0) control_ftol = ftol;
            if(xtol >= 0.0) control_xtol = xtol;
            if(gtol >= 0.0) control_gtol = gtol;
            if(maxcall >= 0) control_maxcall = maxcall;
            if(epsilon >= 0.0) control_epsilon = epsilon;
            if(stepbound >= 0.0) control_stepbound = stepbound;
        }

        FLOAT ** LMfit(int molecules, FLOAT **par0, FLOAT **y)
        {
            n_molecules = molecules;
            TransformForwardParams(par0);

            // working in 1D ... there is only 48kB od shared memory per block, thus with arrays of length 1248,
            // it is possible to run 4 threads with double precision (8B per item) or 8 threads with single precision (4B per item)
            int blockX = 32 / FSIZE;
            int nblocks = (n_molecules / blockX) + (int)(n_molecules % blockX > 0);    // --> ceil(n_molecules / blockX)
            // nmolmem = number of molecules in memory (including the ones used for padding)
            int nmolmem = n_molecules;
            if(n_molecules % blockX > 0)
                nmolmem += 8 - (n_molecules % blockX);
	        
            CUDA cuda;
            
            // Calculate the base address of each vector/matrix.
            CUdeviceptr d_Y = cuda.alloc(nmolmem*n_input_data*FSIZE);
            CUdeviceptr d_A = cuda.alloc(nmolmem*n_params*FSIZE);

            // Arrange the data so the GPU can access to them coallesced in warps
            FLOAT *tmpY, *tmpA;
            tmpY = (FLOAT *)malloc(nmolmem*n_input_data*FSIZE);
            tmpA = (FLOAT *)malloc(nmolmem*n_params*FSIZE);
            // Y
            for(int b = 0, idx = 0; b < nblocks; b++)
                for(int i = 0; i < n_input_data; i++)
                    for(int m = b*blockX; m < (b+1)*blockX; m++, idx++)
                        tmpY[idx] = ((m < n_molecules) ? y[m][i] : 0);

            // A
            for(int b = 0, idx = 0; b < nblocks; b++)
                for(int i = 0; i < n_params; i++)
                    for(int m = b*blockX; m < (b+1)*blockX; m++, idx++)
                        tmpA[idx] = ((m < n_molecules) ? par0[m][i] : 0);

            // Initialize host array and copy it to CUDA device
            cuMemcpyHtoD(d_Y, tmpY, nmolmem*n_input_data*FSIZE);
            cuMemcpyHtoD(d_A, tmpA, nmolmem*n_params*FSIZE);

            // Do calculation on device (this goes through the modified legacy interface)
            int maxcall = control_maxcall * (n_params + 1), one = 1;
            void *args[] = { &n_molecules, &n_boxsize, &d_Y, &n_params, &d_A, &control_ftol, &control_xtol,
                             &control_gtol, &maxcall, &control_epsilon, &one, &control_stepbound };

            // run
            cuda.run("bin/lmmin.ptx", "lmmin", dim3(blockX,1,1), dim3(nblocks,1,1), args);

            // Retrieve result from device and store it in host array
            cuMemcpyDtoH(tmpA, d_A, nmolmem*n_params*FSIZE);
            
            FLOAT **results = (FLOAT**)malloc(n_molecules*sizeof(FLOAT*));
            for(int m = 0; m < n_molecules; m++)
                results[m] = (FLOAT*)malloc(DATAA_SIZE*FSIZE);
            
            for(int b = 0, idx = 0; b < nblocks; b++)
                for(int i = 0; i < n_params; i++)
                    for(int m = b*blockX; m < (b+1)*blockX; m++, idx++)
                        if(m < n_molecules)
                            results[m][i] = tmpA[idx];
    
	        // Clean up
            free(tmpA);
            free(tmpY);
            cuda.free(d_Y);
            cuda.free(d_A);

	        TransformBackParams(results);
            return results;
        }
};

// ------------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------------
template<typename T> inline T SQR(T x)
{
    return x*x;
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

int main()
{
    const int nparams = 5;  // {x,y,I,sigma,bkg}
    const int molecules = 9;
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
    //
    assert(fitregionsize % 2 == 1);
    //
    //
    // initial guesses
	FLOAT **a0 = (FLOAT**)malloc(sizeof(FLOAT*)*molecules);
    for(int i = 0; i < molecules; i++)
    {
        a0[i] = (FLOAT*)malloc(sizeof(FLOAT)*nparams);
        a0[i][0] = 0;
        a0[i][1] = 0;
        a0[i][3] = 1.6f;
        a0[i][4] = min(im,fitregionsize2);
        a0[i][2] = max(im,fitregionsize2) - a0[i][4];
    }
	//
    // Y - raw data
    FLOAT **Y = (FLOAT**)malloc(sizeof(FLOAT*)*molecules);
    for(int i = 0; i < molecules; i++)
    {
        Y[i] = (FLOAT*)malloc(sizeof(FLOAT)*fitregionsize2);
        for(int j = 0; j < fitregionsize2; j++)
            Y[i][j] = im[j];
    }
    //
    // Fitting
    Gaussian2DFitting gauss(5);
    FLOAT **a = gauss.LMfit(molecules, a0, Y);
    //
	// expected results (with double precision): 0.0017, 0.056, 356.73, 0.94857, 182.76
    for(int i = 0; i < molecules; i++)
	    printf("x=%f,y=%f,I=%f,s=%f,b=%f\n",a[i][0],a[i][1],a[i][2],a[i][3],a[i][4]);
    //
	return 0;
}