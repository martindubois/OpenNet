
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDAW.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "CUDAW.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

// CUDAW_Mallot ==> CUDAW_Free
void CUDAW_Free( void * aMemory_DA )
{
    assert( NULL != aMemory_DA );

    // cudaMallog ==> cudaFree  See CUDAW_Malloc
    cudaError_t lRet = cudaFree( aMemory_DA );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaFree(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

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

void CUDAW_GetDeviceProperties( cudaDeviceProp * aProp, int aDevice )
{
    assert( NULL != aProp   );
    assert(    0 <= aDevice );

    cudaError_t lRet = cudaGetDeviceProperties( aProp, aDevice );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaGetDeviceProperties( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUDAW_LaunchKernel( const void * aFunction, dim3 aGridDim, dim3 aBlockDim, void ** aArgs, size_t aSharedMem_byte, cudaStream_t aStream )
{
    assert( NULL != aFunction );
    assert( NULL != aArgs     );
    assert( NULL != aStream   );

    cudaError_t lRet = cudaLaunchKernel( aFunction, aGridDim, aBlockDim, aArgs, aSharedMem_byte, aStream );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaLauchKernel( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// CUDAW_Malloc ==> CUDAW_Free
void CUDAW_Malloc( void * * aMemory_DA, size_t aSize_byte )
{
    assert( NULL != aMemory_DA );
    assert(    0 <  aSize_byte );

    // cudaMalloc ==> cudaFree  See CUDAW_Malloc
    cudaError_t lRet = cudaMalloc( aMemory_DA, aSize_byte );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaGetDeviceCount( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 <= ( * aMemory_DA ) );
}

void CUDAW_SetDevice( int aDevice )
{
    assert( 0 <= aDevice );

    cudaError_t lRet = cudaSetDevice( aDevice );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaSetDevice(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// CUDAW_StreamCreate ==> CUDAW_StreamDestroy
void CUDAW_StreamCreate( cudaStream_t * aStream )
{
    assert( NULL != aStream );

    // cudaStreamCreate ==> cudaStreamDestroy  See CUDAW_StreamDestroy
    cudaError_t lRet = cudaStreamCreate( aStream );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaStreamCreate(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert(NULL != ( * aStream ) );
}

// CUDAW_StreamCreate ==> CUDAW_StreamDestroy
void CUDAW_StreamDestroy( cudaStream_t aStream )
{
    assert( NULL != aStream );

    // cudaStreamCrreate ==> cudaStreamDestroy  See CUDAW_StreamCreate
    cudaError_t lRet = cudaStreamDestroy( aStream );
    if ( cudaSuccess != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cudaStreamDestroy(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}
