
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDAW.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== CUDA ===============================================================
#include <cuda_runtime.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "CUDAW.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void CUDAW_GetDeviceCount( int * aCount )
{
    assert( NULL != aCount );

    cudaError_t lRet = cudaGetDeviceCount( aCount );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaGetDeviceCount(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 <= ( * aCount ) );
}
