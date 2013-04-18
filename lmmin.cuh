#ifndef __LMMIN_CUH__
#define __LMMIN_CUH__

// precision: float/double ...float is usually at least 8x faster than double, because number of double processing units is limited on the chip!
#define DOUBLE_PRECISION 1

#if DOUBLE_PRECISION == 1
    #define LM_MACHEP      DBL_EPSILON   /* resolution of arithmetic */
    #define LM_DWARF       DBL_MIN       /* smallest nonzero number */
    #define LM_SQRT_DWARF  sqrt(DBL_MIN) /* square should not underflow */
    #define LM_SQRT_GIANT  sqrt(DBL_MAX) /* square should not overflow */
    #define FLOAT          double
    #define CUDA_BANK_SIZE cudaSharedMemBankSizeEightByte
#else
    #define LM_MACHEP      FLT_EPSILON   /* resolution of arithmetic */
    #define LM_DWARF       FLT_MIN       /* smallest nonzero number */
    #define LM_SQRT_DWARF  sqrt(FLT_MIN) /* square should not underflow */
    #define LM_SQRT_GIANT  sqrt(FLT_MAX) /* square should not overflow */
    #define FLOAT          float
    #define CUDA_BANK_SIZE cudaSharedMemBankSizeFourByte
#endif

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

#define FSIZE sizeof(FLOAT)

#define DATAX_SIZE (2 * n_input_data)
#define DATAY_SIZE n_input_data
#define DATAA_SIZE n_params
#define FVEC_SIZE n_input_data
#define DIAG_SIZE n_params
#define QTF_SIZE n_params
#define FJAC_SIZE (n_params * n_input_data)
#define WA1_SIZE n_params
#define WA2_SIZE n_params
#define WA3_SIZE n_params
#define WA4_SIZE n_input_data
#define IPVT_SIZE n_params

#define BLOCK_SIZE (DATAA_SIZE + DATAY_SIZE + DATAA_SIZE + FVEC_SIZE + DIAG_SIZE + QTF_SIZE + FJAC_SIZE + WA1_SIZE + WA2_SIZE + WA3_SIZE + WA4_SIZE + IPVT_SIZE)
// --> first DATAA_SIZE is there for the `p` in function `evaluate`

#endif // __LMMIN_CUH__
