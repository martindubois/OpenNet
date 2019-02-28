
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUW.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "CUW.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void CUW_CtxCreate( CUcontext * aContext, unsigned int aFlags, CUdevice aDevice )
{
    assert( NULL != aContext );
    assert(    0 <= aDevice  );

    // cuCtxCreate ==> cuCtxDestroy
    CUresult lRet = cuCtxCreate( aContext, aFlags, aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuCtxCreate( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 != ( * aContext ) );
}

void CUW_CtxDestroy( CUcontext aContext )
{
    assert( NULL != aContext );

    // cuCtxCreate ==> cuCtxDestroy
    CUresult lRet = cuCtxDestroy( aContext );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuCtxDestroy(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_CtxPopCurrent( CUcontext * aContext )
{
    assert( NULL != aContext );

    // cuCtxPushCurrent ==> cuCtxPopCurrent
    CUresult lRet = cuCtxPopCurrent( aContext );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuCtxPopCurrent(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( NULL != ( * aContext ) );
}

void CUW_CtxPushCurrent( CUcontext aContext )
{
    assert( NULL != aContext );

    // cuCtxPushCurrent ==> cuCtxPopCurrent
    CUresult lRet = cuCtxPushCurrent( aContext );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuCtxPushCurrent(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_CtxSetCurrent( CUcontext aContext )
{
    assert( NULL != aContext );

    CUresult lRet = cuCtxSetCurrent( aContext );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuCtxSetCurrent(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_DeviceGet( CUdevice * aDevice, int aIndex )
{
    assert( NULL != aDevice );
    assert(    0 <= aIndex  );

    CUresult lRet = cuDeviceGet( aDevice, aIndex );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDeviceGet( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 <= ( * aDevice ) );
}

void CUW_DeviceGetAttribute( int * aValue, CUdevice_attribute aAttribute, CUdevice aDevice )
{
    assert( NULL != aValue  );
    assert(    0 <= aDevice );

    CUresult lRet = cuDeviceGetAttribute( aValue, aAttribute, aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDeviceGetAttribute( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_DeviceGetCount( int * aCount )
{
    assert( NULL != aCount );

    CUresult lRet = cuDeviceGetCount( aCount );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDeviceGetCount(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 <= ( * aCount ) );
}

void CUW_DeviceGetName( char * aName, int aSize_byte, CUdevice aDevice )
{
    assert( NULL != aName      );
    assert(    0 <  aSize_byte );
    assert(    0 <= aDevice    );

    CUresult lRet = cuDeviceGetName( aName, aSize_byte, aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDeviceGetName( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_DevicePrimaryCtxRelease( CUdevice aDevice )
{
    assert( 0 <= aDevice );

    // cuDevicePrimaryCtxRetain ==> cuDevicePrimaryCtxRelease
    CUresult lRet = cuDevicePrimaryCtxRelease( aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDevicePrimaryCtxRelease(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_DevicePrimaryCtxRetain ( CUcontext * aContext, CUdevice aDevice )
{
    assert( NULL != aContext );
    assert(    0 <= aDevice  );

    // cuDevicePrimaryCtxRetain ==> cuDevicePrimaryCtxRelease
    CUresult lRet = cuDevicePrimaryCtxRetain( aContext, aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDevicePrimaryCtxRetain(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( NULL != ( * aContext ) );
}

void CUW_DeviceTotalMem( size_t * aSize_byte, CUdevice aDevice )
{
    assert( NULL != aSize_byte );
    assert(    0 <= aDevice    );

    CUresult lRet = cuDeviceTotalMem( aSize_byte, aDevice );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuDeviceTotalMem( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 < ( * aSize_byte ) );
}

void CUW_Init( unsigned int aFlags )
{
    CUresult lRet = cuInit( aFlags );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuInit(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_LaunchHostFunction ( CUstream aStream, CUhostFn aFunction, void * aUserData )
{
    assert( NULL != aStream   );
    assert( NULL != aFunction );

    CUresult lRet = cuLaunchHostFunc( aStream, aFunction, aUserData );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuLaunchHostFunc( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_LaunchKernel( CUfunction aFunction, unsigned int aGridDimX, unsigned int aGridDimY, unsigned int aGridDimZ, unsigned int aBlockDimX, unsigned int aBlockDimY, unsigned int aBlockDimZ, unsigned int aSharedMemSize_byte, CUstream aStream, void * * aArguments, void * * aExtra )
{
    // printf( "CUW_LaunchKernel( %p, %u, %u, %u, %u, %u, %u, %u bytes, %p, %p, %p )\n", aFunction, aGridDimX, aGridDimY, aGridDimZ, aBlockDimX, aBlockDimY, aBlockDimZ, aSharedMemSize_byte, aStream, aArguments, aExtra );

    assert( NULL != aFunction  );
    assert(    0 <  aGridDimX  );
    assert(    0 <  aGridDimY  );
    assert(    0 <  aGridDimZ  );
    assert(    0 <  aBlockDimX );
    assert(    0 <  aBlockDimY );
    assert(    0 <  aBlockDimZ );
    assert( NULL != aStream    );

    CUresult lRet = cuLaunchKernel( aFunction, aGridDimX, aGridDimY, aGridDimZ, aBlockDimX, aBlockDimY, aBlockDimZ, aSharedMemSize_byte, aStream, aArguments, aExtra );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuLaunchKernel( , , , , , , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_MemAlloc( CUdeviceptr * aPtr_DA, size_t aSize_byte )
{
    assert( NULL != aPtr_DA    );
    assert(    0 <  aSize_byte );

    // cuMemAlloc ==> cuMemFree
    CUresult lRet = cuMemAlloc( aPtr_DA, aSize_byte );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuMemAlloc( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 != aPtr_DA );
}

void CUW_MemFree( CUdeviceptr aPtr_DA )
{
    assert( 0 != aPtr_DA );

    // cuMemAlloc ==> cuMemFree
    CUresult lRet = cuMemFree( aPtr_DA );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuMemFree(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_ModuleGetFunction( CUfunction * aFunction, CUmodule aModule, const char * aName )
{
    assert( NULL != aFunction );
    assert( NULL != aModule   );
    assert( NULL != aName     );

    CUresult lRet = cuModuleGetFunction( aFunction, aModule, aName );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuModuleGetFunction( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( NULL != ( * aFunction ) );

}

void CUW_ModuleLoadDataEx( CUmodule * aModule, const void * aImage, unsigned int aNumOptions, CUjit_option * aOptions, void * * aOptionValues )
{
    assert( NULL != aModule );
    assert( NULL != aImage  );

    // cuModuleLoadDataEx ==> cuModuleUnload
    CUresult lRet = cuModuleLoadDataEx( aModule, aImage, aNumOptions, aOptions, aOptionValues );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuModuleLoadDataEx( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( NULL != ( * aModule ) );
}

void CUW_ModuleUnload( CUmodule aModule )
{
    assert( NULL != aModule );

    // cuModuleLoadDataEx ==> cuModuleUnload
    CUresult lRet = cuModuleUnload( aModule );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuModuleUnload(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_PointerSetAttribute( const void * aValue, CUpointer_attribute aAttribute, CUdeviceptr aPtr_DA )
{
    assert( NULL != aValue  );
    assert(    0 != aPtr_DA );

    CUresult lRet = cuPointerSetAttribute( aValue, aAttribute, aPtr_DA );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuPointerSetAttribute( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void CUW_StreamCreate( CUstream * aStream, unsigned int aFlags )
{
    assert( NULL != aStream );

    // cuStreamCreate ==> cuStreamDestroy
    CUresult lRet = cuStreamCreate( aStream, aFlags );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuStreamCreate( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( NULL != ( * aStream ) );
}

void CUW_StreamDestroy( CUstream aStream )
{
    assert( NULL != aStream );

    // cuStreamCreate ==> cuStreamDestroy
    CUresult lRet = cuStreamDestroy( aStream );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuStreamDestroy( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}
