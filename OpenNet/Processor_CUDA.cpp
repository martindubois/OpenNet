
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_CUDA.cpp

#define __CLASS__ "Processor_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNetK/Types.h>

// ===== OpenNet ============================================================
#include "Buffer_Data_CUDA.h"
#include "CUW.h"
#include "NVRTCW.h"
#include "Thread_Functions_CUDA.h"

#include "Processor_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aDevice
// aDebugLog [-K-;RW-]
//
// Exception  KmsLib::Exception *  See InitInfo
// Threads  Apps
Processor_CUDA::Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog )
    : Processor_Internal( aDebugLog )
{
    assert(    0 <= aDevice   );
    assert( NULL != aDebugLog );

    CUW_DeviceGet( & mDevice, aDevice );

    // CUW_CtxCreate ==> CUW_CtxDestroy  See the destructor
    CUW_CtxCreate( & mContext, 0, mDevice );

    InitInfo();
}

// aPacketSize_byte
// aBuffer [---;RW-]
//
// Return  This function return the created instance
Buffer_Data * Processor_CUDA::Buffer_Allocate(unsigned int aPacketSize_byte, OpenNetK::Buffer * aBuffer)
{
    // printf( __CLASS__ "Buffer_Allocate( %u byte, 0x%p )\n", aPacketSize_byte, aBuffer );

    assert(    0 <  aPacketSize_byte );
    assert( NULL != aBuffer          );

    assert( 0 <= mDevice );

    int lPacketQty;

    CUW_DeviceGetAttribute( & lPacketQty, CU_DEVICE_ATTRIBUTE_WARP_SIZE, mDevice );

    aBuffer->mSize_byte = sizeof(OpenNet_BufferHeader);

    aBuffer->mSize_byte += sizeof(OpenNet_PacketInfo) * static_cast<unsigned int>(lPacketQty);
    aBuffer->mSize_byte += aPacketSize_byte           * static_cast<unsigned int>(lPacketQty);
    aBuffer->mSize_byte += (aBuffer->mSize_byte / OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) * aPacketSize_byte;

    unsigned int lMod_byte = aBuffer->mSize_byte % 0x10000;
    if ( 0 != lMod_byte )
    {
        aBuffer->mSize_byte += 0x10000 - lMod_byte;
    }

    CUdeviceptr lMem_DA;

    // CUW_MemAlloc ==> CUW_MemFree  See Buffer_Data_CUDA::~Buffer_Data_CUDA
    CUW_MemAlloc( & lMem_DA, aBuffer->mSize_byte );
    assert( 0 != lMem_DA );

    unsigned int lFlag = 1;

    CUW_PointerSetAttribute( & lFlag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, reinterpret_cast< CUdeviceptr >( lMem_DA ) );

    // new ==> delete
    Buffer_Data * lResult = new Buffer_Data_CUDA( lMem_DA, lPacketQty );

    aBuffer->mBuffer_DA = lMem_DA;
    aBuffer->mBuffer_PA = 0;
    aBuffer->mMarker_PA = 0;
    aBuffer->mPacketQty = static_cast<uint32_t>(lPacketQty);

    return lResult;
}

// aKernel [---;R--] The kernel to compile
//
// Return  The new CUmodule instance
//
// Exception  KmsLib::Exception *  See Program_CreateAndCompile,
//                                 NVRTCW_GetPTXSize, NVRTCW_GetPTX,
//                                 NVRTCW_DestroyProgram and
//                                 NVRTCW_ModuleLoadDataEx
//
// Processor_CUDA::Module_Create ==> NVRTCW_ModuleUnload
CUmodule Processor_CUDA::Module_Create( OpenNet::Kernel * aKernel )
{
    assert( NULL != aKernel );

    assert(NULL != mDebugLog);
    assert(   0 <= mDevice  );

    nvrtcProgram lProgram = Program_CreateAndCompile( aKernel );

    size_t lSize_byte;

    NVRTCW_GetPTXSize( lProgram, & lSize_byte );
    assert( 0 < lSize_byte );

    char * lPTX = new char [ lSize_byte ];
    assert( NULL != lPTX );

    try
    {
        NVRTCW_GetPTX( lProgram, lPTX );
    }
    catch ( ... )
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Module_Create", __LINE__);

        NVRTCW_DestroyProgram( & lProgram );

        // printf( __CLASS__ "Module_Create - delete [] 0x%lx (lPTX)\n", reinterpret_cast< uint64_t >( lPTX ) );

        delete [] lPTX;

        throw;
    }

    CUmodule lResult;

    try
    {
        NVRTCW_DestroyProgram( & lProgram );

        // NVRTCW_ModuleLoadDataEx ==> NVRTC_ModuleUnload
        CUW_ModuleLoadDataEx( & lResult, lPTX, 0, 0, 0 );
    }
    catch ( ... )
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Module_Create", __LINE__);

        // printf( __CLASS__ "Module_Create - delete [] 0x%lx (lPTX)\n", reinterpret_cast< uint64_t >( lPTX ) );

        delete [] lPTX;

        throw;
    }

    // printf( __CLASS__ "Module_Create - delete [] 0x%lx (LTX)\n", reinterpret_cast< uint64_t >( lPTX ) );

    delete [] lPTX;

    return lResult;
}

// ===== Processor_Internal =================================================

Thread_Functions * Processor_CUDA::Thread_Get()
{
    assert(NULL != mDebugLog);

    if (NULL == mThread)
    {
        // new ==> delete
        mThread = new Thread_Functions_CUDA(this, mConfig.mFlags.mProfilingEnabled, mDebugLog);
        assert(NULL != mThread);
    }

    return mThread;
}

// ===== OpenNet::Processor =================================================

Processor_CUDA::~Processor_CUDA()
{
    // printf( __CLASS__ "~Processor_CUDA()\n" );

    assert( NULL != mContext );

    // CUW_CtxCreate ==> CUW_CtxDestroy  See the constructor
    CUW_CtxDestroy( mContext );
}

void * Processor_CUDA::GetContext()
{
    assert( NULL != mContext );

    return mContext;
}

void * Processor_CUDA::GetDevice()
{
    assert( 0 <= mDevice );

    return (void *)( static_cast< uint64_t >( mDevice ) );
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Processor_CUDA::InitInfo()
{
    assert( 0 <= mDevice );

    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mGlobalMemCacheSize_byte    ) , CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE              , mDevice );
    CUW_DeviceTotalMem    ( & mInfo.mGlobalMemSize_byte                                                                                       , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mImage2DMaxHeight           ) , CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT   , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mImage2DMaxWidth            ) , CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH    , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mImage3DMaxDepth            ) , CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH    , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mImage3DMaxHeight           ) , CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT   , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mImage3DMaxWidth            ) , CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH    , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mLocalMemSize_byte          ) , CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mMaxConstantBufferSize_byte ) , CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY      , mDevice );
    // CUW_DeviceGetAttribute( & mInfo.mMaxMemAllocSize_byte
    // CUW_DeviceGetAttribute( & mInfo.mMaxParameterSize_byte
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mMaxWorkGroupSize           ) , CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X             , mDevice );
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mMaxWorkItemSizes           ) , CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X            , mDevice );

    // CUW_DeviceGetAttribute( & mInfo.mGlobalMemCacheType
    // CUW_DeviceGetAttribute( & mInfo.mGlobalMemCacheLineSize_byte
    // CUW_DeviceGetAttribute( & mInfo.mLocalMemType
    CUW_DeviceGetAttribute( reinterpret_cast< int * >( & mInfo.mMaxComputeUnits            ) , CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT       , mDevice );
    // CUW_DeviceGetAttribute( & mInfo.mMaxConstantArgs            , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mMaxReadImageArgs           , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mMaxSamplers                , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mMaxWriteImageArgs          , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mMemBaseAddrAlign_bit       , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mMinDataTypeAlignSize_byte  , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mPreferredVectorWidthChar   , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mPreferredVectorWidthShort  , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mPreferredVectorWidthInt    , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mPreferredVectorWidthLong   , CU_DEVICE_ATTRIBUTE_
    // CUW_DeviceGetAttribute( & mInfo.mVendorId                   , CU_DEVICE_ATTRIBUTE_

    mInfo.mFlags.mAvailable         = true;
    mInfo.mFlags.mCompilerAvailable = true;
    mInfo.mFlags.mEndianLittle      = true;
    mInfo.mFlags.mImageSupport      = true;

    // GetDeviceInfo(CL_DRIVER_VERSION, sizeof(mInfo.mDriverVersion), &mInfo.mDriverVersion);
    CUW_DeviceGetName( mInfo.mName, sizeof( mInfo.mName ), mDevice );
    // GetDeviceInfo(CL_DEVICE_PROFILE, sizeof(mInfo.mProfile      ), &mInfo.mProfile      );
    // GetDeviceInfo(CL_DEVICE_VENDOR , sizeof(mInfo.mVendor       ), &mInfo.mVendor       );
    // GetDeviceInfo(CL_DEVICE_VERSION, sizeof(mInfo.mVersion      ), &mInfo.mVersion      );
}

// aKernel [---;R--] The kernel to compile
//
// Return  The new nvrtcProgram instance
//
// Exception  KmsLib::Exception *  See NVRTCW_CreateProgram,
//                                 VRTCW_GetProgramLogSize,
//                                 NVRTCW_GetProgramLog and
//                                 NVRTC_DestroyProgram
//
// Process_CUDA::Program_CreateAndCompile ==> NVRTCW_DestroyProgram
nvrtcProgram Processor_CUDA::Program_CreateAndCompile( OpenNet::Kernel * aKernel )
{
    assert( NULL != aKernel );

    assert( NULL != mDebugLog );

    nvrtcProgram lResult;

    // NVRTCW_CreateProgram ==> NVRTCW_DestroyProgram
    NVRTCW_CreateProgram( & lResult, aKernel->GetCode(), NULL, 0, NULL, NULL );
    assert( NULL != lResult );

    const char * lOptions[ 2 ];

    lOptions[ 0 ] = "-I /home/mdubois/OpenNet/Includes";
    lOptions[ 1 ] = "-D_OPEN_NET_CUDA_";

    try
    {
        NVRTCW_CompileProgram( lResult, sizeof( lOptions ) / sizeof( lOptions[ 0 ] ), lOptions );
    }
    catch ( KmsLib::Exception * eE )
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Program_CreateAndCompile", __LINE__);
        mDebugLog->Log( eE );

        size_t lSize_byte;

        NVRTCW_GetProgramLogSize(   lResult, & lSize_byte );
        NVRTCW_GetProgramLog    (   lResult, aKernel->AllocateBuildLog( lSize_byte ) );
        NVRTCW_DestroyProgram   ( & lResult );

        throw;
    }

    return lResult;
}
