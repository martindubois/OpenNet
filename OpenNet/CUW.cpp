
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUW.cpp

// CONFIG  _CHECK_
//         Define to enable basic leak check
#define _CHECK_

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "CUW.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define CHECK_CTX_CREATE                (0)
#define CHECK_DEVICE_PRIMARY_CTX_RETAIN (1)
#define CHECK_EVENT_CREATE              (2)
#define CHECK_MEM_ALLOC                 (3)
#define CHECK_MODULE_LOAD_DATA_EX       (4)
#define CHECK_STREAM_CREATE             (5)

#define CHECK_QTY (6)

// Static variable
/////////////////////////////////////////////////////////////////////////////

#ifdef _CHECK_
    static unsigned int sCheck[ CHECK_QTY ][ 2 ];
#endif

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef _CHECK_
    #define CHECK(C,D) sCheck[(C)][(D)] ++;
#else
    #define CHECK(C,D)
#endif

// Functions
/////////////////////////////////////////////////////////////////////////////

void CUW_Check()
{
    #ifdef _CHECK_

        for ( unsigned int i = 0; i < CHECK_QTY; i ++ )
        {
            if ( sCheck[ i ][ 0 ] != sCheck[ i ][ 1 ] )
            {
                printf( "Cheak %u failed - %u != %u\n", i, sCheck[ i ][ 0 ], sCheck[ i ][ 1 ] );
            }
        }

    #endif
}

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

    CHECK( CHECK_CTX_CREATE, 0 );

    assert( 0 != ( * aContext ) );
}

void CUW_CtxDestroy( CUcontext aContext )
{
    assert( NULL != aContext );

    CHECK( CHECK_CTX_CREATE, 1 );

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

    CHECK( CHECK_DEVICE_PRIMARY_CTX_RETAIN, 1 );

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

    CHECK( CHECK_DEVICE_PRIMARY_CTX_RETAIN, 0 );

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

void CUW_EventCreate( CUevent * aEvent, unsigned int aFlags )
{
    assert( NULL != aEvent );

    // cuEventCreate ==> cuEventDestroy
    CUresult lRet = cuEventCreate( aEvent, aFlags );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuEventCreate(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    CHECK( CHECK_EVENT_CREATE, 0 );

    assert( NULL != ( * aEvent ) );
}

void CUW_EventDestroy( CUevent aEvent )
{
    assert( NULL != aEvent );

    CHECK( CHECK_EVENT_CREATE, 1 );

    // cuEventCreate ==> cuEventDestroy
    CUresult lRet = cuEventDestroy( aEvent );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuEventDestroy(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
void CUW_EventElapsedTime( float * aElapsed_ms, CUevent aStart, CUevent aEnd )
{
    assert( NULL != aElapsed_ms );
    assert( NULL != aStart      );
    assert( NULL != aEnd        );

    CUresult lRet = cuEventElapsedTime( aElapsed_ms, aStart, aEnd );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuEventElapsedTime( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// CRITICAL PATH  Processing
//                1 / iteration, 1 more when profiling is enabled
void CUW_EventRecord( CUevent aEvent, CUstream aStream )
{
    assert( NULL != aEvent  );
    assert( NULL != aStream );

    CUresult lRet = cuEventRecord( aEvent, aStream );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuEventRecord( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// CRITICAL PATH  Processing
//                1 / iteration
void CUW_EventSynchronize( CUevent aEvent )
{
    assert( NULL != aEvent );

    CUresult lRet = cuEventSynchronize( aEvent );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuEventSynchronize(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
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

void CUW_LaunchHostFunc( CUstream aStream, CUhostFn aFunction, void * aUserData )
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

// CRITICAL PATH  Processing
//                1 / iteration
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

    CHECK( CHECK_MEM_ALLOC, 0 );

    assert( 0 != aPtr_DA );
}

void CUW_MemFree( CUdeviceptr aPtr_DA )
{
    assert( 0 != aPtr_DA );

    CHECK( CHECK_MEM_ALLOC, 1 );

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

    CHECK( CHECK_MODULE_LOAD_DATA_EX, 0 );

    assert( NULL != ( * aModule ) );
}

void CUW_ModuleUnload( CUmodule aModule )
{
    assert( NULL != aModule );

    CHECK( CHECK_MODULE_LOAD_DATA_EX, 1 );

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

    CHECK( CHECK_STREAM_CREATE, 0 );

    assert( NULL != ( * aStream ) );
}

void CUW_StreamDestroy( CUstream aStream )
{
    assert( NULL != aStream );

    CHECK( CHECK_STREAM_CREATE, 1 );

    // cuStreamCreate ==> cuStreamDestroy
    CUresult lRet = cuStreamDestroy( aStream );
    if ( CUDA_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "cuStreamDestroy( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}
