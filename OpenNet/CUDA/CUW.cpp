
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/CUW.cpp

// CONFIG  _CHECK_
//         Define to enable basic leak check
#define _CHECK_

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet/CUDA =======================================================
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

#define VERIFY_RET(M)                                                 \
    if ( CUDA_SUCCESS != lRet )                                       \
    {                                                                 \
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN, \
            (M), NULL, __FILE__, __FUNCTION__, __LINE__, lRet );      \
    }

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

void CUW_DeviceGetAttribute( int * aValue, CUdevice_attribute aAttribute, CUdevice aDevice )
{
    assert( NULL != aValue  );
    assert(    0 <= aDevice );

    CUresult lRet = cuDeviceGetAttribute( aValue, aAttribute, aDevice );
    VERIFY_RET( "cuDeviceGetAttribute( , ,  ) failed" );
}

void CUW_DeviceGetCount( int * aCount )
{
    assert( NULL != aCount );

    CUresult lRet = cuDeviceGetCount( aCount );
    VERIFY_RET( "cuDeviceGetCount(  ) failed" );

    assert( 0 <= ( * aCount ) );
}

void CUW_DeviceGetName( char * aName, int aSize_byte, CUdevice aDevice )
{
    assert( NULL != aName      );
    assert(    0 <  aSize_byte );
    assert(    0 <= aDevice    );

    CUresult lRet = cuDeviceGetName( aName, aSize_byte, aDevice );
    VERIFY_RET( "cuDeviceGetName( , ,  ) failed" );
}

void CUW_DeviceTotalMem( size_t * aSize_byte, CUdevice aDevice )
{
    assert( NULL != aSize_byte );
    assert(    0 <= aDevice    );

    CUresult lRet = cuDeviceTotalMem( aSize_byte, aDevice );
    VERIFY_RET( "cuDeviceTotalMem( ,  ) failed" );

    assert( 0 < ( * aSize_byte ) );
}

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
void CUW_EventElapsedTime( float * aElapsed_ms, CUevent aStart, CUevent aEnd )
{
    assert( NULL != aElapsed_ms );
    assert( NULL != aStart      );
    assert( NULL != aEnd        );

    CUresult lRet = cuEventElapsedTime( aElapsed_ms, aStart, aEnd );
    VERIFY_RET( "cuEventElapsedTime( , ,  ) failed" );
}

void CUW_Init( unsigned int aFlags )
{
    CUresult lRet = cuInit( aFlags );
    VERIFY_RET( "cuInit(  ) failed" );
}

void CUW_MemcpyDtoH( void * aDst, CUdeviceptr aSrc, size_t aSize_byte )
{
    assert( NULL != aDst       );
    assert(    0 != aSrc       );
    assert(    0 <  aSize_byte );

    CUresult lRet = cuMemcpyDtoH( aDst, aSrc, aSize_byte );
    VERIFY_RET( "cuMemcpyDtoH( , ,  ) failed" );
}

// ===== CUcontext ==========================================================

void CUW_CtxCreate( CUcontext * aContext, unsigned int aFlags, CUdevice aDevice )
{
    assert( NULL != aContext );
    assert(    0 <= aDevice  );

    // cuCtxCreate ==> cuCtxDestroy
    CUresult lRet = cuCtxCreate( aContext, aFlags, aDevice );
    VERIFY_RET( "cuCtxCreate( , ,  ) failed" );

    CHECK( CHECK_CTX_CREATE, 0 );

    assert( 0 != ( * aContext ) );
}

void CUW_CtxDestroy( CUcontext aContext )
{
    assert( NULL != aContext );

    CHECK( CHECK_CTX_CREATE, 1 );

    // cuCtxCreate ==> cuCtxDestroy
    CUresult lRet = cuCtxDestroy( aContext );
    VERIFY_RET( "cuCtxDestroy(  ) failed" );
}

void CUW_CtxPopCurrent( CUcontext * aContext )
{
    assert( NULL != aContext );

    // cuCtxPushCurrent ==> cuCtxPopCurrent
    CUresult lRet = cuCtxPopCurrent( aContext );
    VERIFY_RET( "cuCtxPopCurrent(  ) failed" );

    assert( NULL != ( * aContext ) );
}

void CUW_CtxPushCurrent( CUcontext aContext )
{
    assert( NULL != aContext );

    // cuCtxPushCurrent ==> cuCtxPopCurrent
    CUresult lRet = cuCtxPushCurrent( aContext );
    VERIFY_RET( "cuCtxPushCurrent(  ) failed" );
}

void CUW_CtxSetCurrent( CUcontext aContext )
{
    assert( NULL != aContext );

    CUresult lRet = cuCtxSetCurrent( aContext );
    VERIFY_RET( "cuCtxSetCurrent(  ) failed" );
}

void CUW_DevicePrimaryCtxRetain ( CUcontext * aContext, CUdevice aDevice )
{
    assert( NULL != aContext );
    assert(    0 <= aDevice  );

    // cuDevicePrimaryCtxRetain ==> cuDevicePrimaryCtxRelease
    CUresult lRet = cuDevicePrimaryCtxRetain( aContext, aDevice );
    VERIFY_RET( "cuDevicePrimaryCtxRetain(  ) failed" );

    CHECK( CHECK_DEVICE_PRIMARY_CTX_RETAIN, 0 );

    assert( NULL != ( * aContext ) );
}

// ===== CUdevice ===========================================================

void CUW_DeviceGet( CUdevice * aDevice, int aIndex )
{
    assert( NULL != aDevice );
    assert(    0 <= aIndex  );

    CUresult lRet = cuDeviceGet( aDevice, aIndex );
    VERIFY_RET( "cuDeviceGet( ,  ) failed" );

    assert( 0 <= ( * aDevice ) );
}

void CUW_DevicePrimaryCtxRelease( CUdevice aDevice )
{
    assert( 0 <= aDevice );

    CHECK( CHECK_DEVICE_PRIMARY_CTX_RETAIN, 1 );

    // cuDevicePrimaryCtxRetain ==> cuDevicePrimaryCtxRelease
    CUresult lRet = cuDevicePrimaryCtxRelease( aDevice );
    VERIFY_RET( "cuDevicePrimaryCtxRelease(  ) failed" );
}

// ===== CUdeviceptr ========================================================

void CUW_MemAlloc( CUdeviceptr * aPtr_DA, size_t aSize_byte )
{
    assert( NULL != aPtr_DA    );
    assert(    0 <  aSize_byte );

    // cuMemAlloc ==> cuMemFree
    CUresult lRet = cuMemAlloc( aPtr_DA, aSize_byte );
    VERIFY_RET( "cuMemAlloc( ,  ) failed" );

    CHECK( CHECK_MEM_ALLOC, 0 );

    assert( 0 != aPtr_DA );
}

void CUW_MemcpyHtoD( CUdeviceptr aDst_DA, const void * aSrc, size_t aSize_byte )
{
    assert(    0 != aDst_DA    );
    assert( NULL != aSrc       );
    assert(    0 <  aSize_byte );

    CUresult lRet = cuMemcpyHtoD( aDst_DA, aSrc, aSize_byte );
    VERIFY_RET( "cuMemcpyHtoD( , ,  ) failed" );
}

void CUW_MemFree( CUdeviceptr aPtr_DA )
{
    assert( 0 != aPtr_DA );

    CHECK( CHECK_MEM_ALLOC, 1 );

    // cuMemAlloc ==> cuMemFree
    CUresult lRet = cuMemFree( aPtr_DA );
    VERIFY_RET( "cuMemFree(  ) failed" );
}

void CUW_MemsetD8( CUdeviceptr aPtr_DA, unsigned char aValue, size_t aSize_byte )
{
    assert( 0 != aPtr_DA    );
    assert( 0 <  aSize_byte );

    CUresult lRet = cuMemsetD8( aPtr_DA, aValue, aSize_byte );
    VERIFY_RET( "cuMemsetD8( , ,  ) failed" );
}

// ===== CUevent ============================================================

void CUW_EventCreate( CUevent * aEvent, unsigned int aFlags )
{
    assert( NULL != aEvent );

    // cuEventCreate ==> cuEventDestroy
    CUresult lRet = cuEventCreate( aEvent, aFlags );
    VERIFY_RET( "cuEventCreate(  ) failed" );

    CHECK( CHECK_EVENT_CREATE, 0 );

    assert( NULL != ( * aEvent ) );
}

void CUW_EventDestroy( CUevent aEvent )
{
    assert( NULL != aEvent );

    CHECK( CHECK_EVENT_CREATE, 1 );

    // cuEventCreate ==> cuEventDestroy
    CUresult lRet = cuEventDestroy( aEvent );
    VERIFY_RET( "cuEventDestroy(  ) failed" );
}

// CRITICAL PATH  Processing
//                1 / iteration, 1 more when profiling is enabled
void CUW_EventRecord( CUevent aEvent, CUstream aStream )
{
    assert( NULL != aEvent  );
    assert( NULL != aStream );

    CUresult lRet = cuEventRecord( aEvent, aStream );
    VERIFY_RET( "cuEventRecord( ,  ) failed" );
}

// CRITICAL PATH  Processing
//                1 / iteration
void CUW_EventSynchronize( CUevent aEvent )
{
    assert( NULL != aEvent );

    CUresult lRet = cuEventSynchronize( aEvent );
    VERIFY_RET( "cuEventSynchronize(  ) failed" );
}

// ===== CUfunction =========================================================

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
    VERIFY_RET( "cuLaunchKernel( , , , , , , , , , ,  ) failed" );
}

void CUW_ModuleGetFunction( CUfunction * aFunction, CUmodule aModule, const char * aName )
{
    assert( NULL != aFunction );
    assert( NULL != aModule   );
    assert( NULL != aName     );

    CUresult lRet = cuModuleGetFunction( aFunction, aModule, aName );
    VERIFY_RET( "cuModuleGetFunction( , ,  ) failed" );

    assert( NULL != ( * aFunction ) );

}

// ===== CUmodule ===========================================================

void CUW_ModuleLoadDataEx( CUmodule * aModule, const void * aImage, unsigned int aNumOptions, CUjit_option * aOptions, void * * aOptionValues )
{
    assert( NULL != aModule );
    assert( NULL != aImage  );

    // cuModuleLoadDataEx ==> cuModuleUnload
    CUresult lRet = cuModuleLoadDataEx( aModule, aImage, aNumOptions, aOptions, aOptionValues );
    VERIFY_RET( "cuModuleLoadDataEx( , , , ,  ) failed" );

    CHECK( CHECK_MODULE_LOAD_DATA_EX, 0 );

    assert( NULL != ( * aModule ) );
}

void CUW_ModuleUnload( CUmodule aModule )
{
    assert( NULL != aModule );

    CHECK( CHECK_MODULE_LOAD_DATA_EX, 1 );

    // cuModuleLoadDataEx ==> cuModuleUnload
    CUresult lRet = cuModuleUnload( aModule );
    VERIFY_RET( "cuModuleUnload(  ) failed" );
}

void CUW_PointerSetAttribute( const void * aValue, CUpointer_attribute aAttribute, CUdeviceptr aPtr_DA )
{
    assert( NULL != aValue  );
    assert(    0 != aPtr_DA );

    CUresult lRet = cuPointerSetAttribute( aValue, aAttribute, aPtr_DA );
    VERIFY_RET( "cuPointerSetAttribute( , ,  ) failed" );
}

// ===== CUstream ===========================================================

void CUW_LaunchHostFunc( CUstream aStream, CUhostFn aFunction, void * aUserData )
{
    assert( NULL != aStream   );
    assert( NULL != aFunction );

    CUresult lRet = cuLaunchHostFunc( aStream, aFunction, aUserData );
    VERIFY_RET( "cuLaunchHostFunc( , ,  ) failed" );
}

void CUW_StreamCreate( CUstream * aStream, unsigned int aFlags )
{
    assert( NULL != aStream );

    // cuStreamCreate ==> cuStreamDestroy
    CUresult lRet = cuStreamCreate( aStream, aFlags );
    VERIFY_RET( "cuStreamCreate( ,  ) failed" );

    CHECK( CHECK_STREAM_CREATE, 0 );

    assert( NULL != ( * aStream ) );
}

void CUW_StreamDestroy( CUstream aStream )
{
    assert( NULL != aStream );

    CHECK( CHECK_STREAM_CREATE, 1 );

    // cuStreamCreate ==> cuStreamDestroy
    CUresult lRet = cuStreamDestroy( aStream );
    VERIFY_RET( "cuStreamDestroy( ,  ) failed" );
}
