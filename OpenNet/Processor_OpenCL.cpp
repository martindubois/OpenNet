
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Types.h>

// ===== OpenNet ============================================================
#include "Buffer_Data_OpenCL.h"
#include "Constants.h"
#include "OCLW.h"
#include "Thread_Functions_OpenCL.h"

#include "Processor_OpenCL.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const cl_queue_properties PROFILING_ENABLED[] =
{
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
    0
};

// Public
/////////////////////////////////////////////////////////////////////////////

// aPlatform
// aDevice
// aDebugLog [-K-;RW-]
//
// Exception  KmsLib::Exception *  See InitInfo
//                                 See OCLW_CreateContext
// Threads  Apps
Processor_OpenCL::Processor_OpenCL(cl_platform_id aPlatform, cl_device_id aDevice, KmsLib::DebugLog * aDebugLog)
    : Processor_Internal(aDebugLog)
    , mDevice( aDevice )
{
    assert(   0 != aPlatform);
    assert(   0 != aDevice  );
    assert(NULL != aDebugLog);

    InitInfo();

    cl_context_properties lProperties[3];

    lProperties[0] = CL_CONTEXT_PLATFORM;
    lProperties[1] = (cl_context_properties)(aPlatform);
    lProperties[2] = 0;

    // OCLW_CreateContext ==> OCLW_ReleaseContext  See ~Processor_OpenCL
    mContext = OCLW_CreateContext(lProperties, 1, &aDevice);
    assert(NULL != mContext);
}

// aPacketSize_byte
// aCommandQueue [---;RW-]
// aKernel       [---;R--]
// aBuffer       [---;-W-]
//
// Return  This method returns a newly created Buffer_Data instance. The
//         caller is responsible for releasing it when it is no longer
//         needed.
//
// Exception  KmsLib::Exception *  CODE_OPEN_CL_ERROR
//                                 See GetKernelWorkGroupInfo
//                                 See OCLW_CreateBuffer
// Threads  Apps
Buffer_Data * Processor_OpenCL::Buffer_Allocate(unsigned int aPacketSize_byte, cl_command_queue aCommandQueue, cl_kernel aKernel, OpenNetK::Buffer * aBuffer)
{
    assert(NULL                 != aCommandQueue   );
    assert(NULL                 != aKernel         );
    assert(NULL                 != aBuffer         );

    assert(NULL != mContext);

    size_t lPacketQty;

    GetKernelWorkGroupInfo(aKernel, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lPacketQty), &lPacketQty);

    aBuffer->mSize_byte = sizeof(OpenNet_BufferHeader);

    aBuffer->mSize_byte += sizeof(OpenNet_PacketInfo) * static_cast<unsigned int>(lPacketQty);
    aBuffer->mSize_byte += aPacketSize_byte           * static_cast<unsigned int>(lPacketQty);
    aBuffer->mSize_byte += (aBuffer->mSize_byte / OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) * aPacketSize_byte;

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See Buffer_Data::Release
    cl_mem lMem = OCLW_CreateBuffer(mContext, CL_MEM_BUS_ADDRESSABLE_AMD, aBuffer->mSize_byte);
    assert(NULL != lMem);

    cl_bus_address_amd lBusAddress;

    OCLW_EnqueueMakeBufferResident(aCommandQueue, 1, &lMem, CL_TRUE, &lBusAddress, 0, NULL, NULL);

    Buffer_Data * lResult = new Buffer_Data_OpenCL(lMem, static_cast<unsigned int>(lPacketQty));

    aBuffer->mBuffer_DA =                               0;
    aBuffer->mBuffer_PA = lBusAddress.surface_bus_address;
    aBuffer->mMarker_PA = lBusAddress.marker_bus_address ;
    aBuffer->mPacketQty = static_cast<uint32_t>(lPacketQty);

    return lResult;
}

// aProfilingEnabled
//
// Return  This method returns a newly created cl_command_queue. The caller
//         is responsible for releasing it when it is no longer needed.
cl_command_queue Processor_OpenCL::CommandQueue_Create(bool aProfilingEnabled)
{
    assert(NULL != mContext);
    assert(   0 != mDevice );

    const cl_queue_properties * lProperties = aProfilingEnabled ? PROFILING_ENABLED : NULL;

    // OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue
    return OCLW_CreateCommandQueueWithProperties(mContext, mDevice, lProperties);
}

// aKernel [---;RW-]
//
// Return  This method returns a newly created cl_program. The caller is
//         responsible for releasing it when it is no longer needed.
cl_program Processor_OpenCL::Program_Create(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert(NULL != mContext );
    assert(NULL != mDebugLog);
    assert(   0 != mDevice  );

    // OCLW_CreateProgramWithSource ==> OCLW_ReleaseProgram
    cl_program lResult = OCLW_CreateProgramWithSource(mContext, aKernel->GetCodeLineCount(), aKernel->GetCodeLines(), NULL);
    assert(NULL != lResult);

    // TODO  OpenNet::Processor_Internal
    //       High (Feature) - Utiliser un technique de recherche pour trouver
    //       le repertoire Includes. Il faut trouver un repertoire qui
    //       contient le fichier "OpenNetK/Kernel.h". Essayer en premier la
    //       variable d'environnement OPEN_NET_INCLUDES. Essayer en dernier
    //       "V:/OpenNet/Includes".
    try
    {
        OCLW_BuildProgram(lResult, 1, &mDevice, "-D _OPEN_NET_OPEN_CL_ -I V:/OpenNet/Includes", NULL, NULL);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        OCLW_GetProgramBuildInfo(lResult, mDevice, CL_PROGRAM_BUILD_LOG, BUILD_LOG_MAX_SIZE_byte, aKernel->AllocateBuildLog());

        OCLW_ReleaseProgram(lResult);
        throw;
    }

    return lResult;
}

// ===== Processor_Internal =================================================

Thread_Functions * Processor_OpenCL::Thread_Get()
{
    assert(NULL != mDebugLog);

    if (NULL == mThread)
    {
        mThread = new Thread_Functions_OpenCL(this, mConfig.mFlags.mProfilingEnabled, mDebugLog);
        assert(NULL != mThread);
    }

    return mThread;
}

// ===== OpenNet::Processor =================================================

// Exception  KmsLib::Exception *  See OCL_ReleaseContext
// Threads  Apps
Processor_OpenCL::~Processor_OpenCL()
{
    assert(NULL != mContext);

    // OCLW_CreateContext ==> OCLW_ReleaseContext  See Processor_OpenCL
    OCLW_ReleaseContext(mContext);
}

void * Processor_OpenCL::GetContext()
{
    assert(NULL != mContext);

    return mContext;
}

void * Processor_OpenCL::GetDevice()
{
    assert(NULL != mDevice);

    return mDevice;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  See GetDeviceInfo
// Threads  Apps
void Processor_OpenCL::InitInfo()
{
    GetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE   , sizeof(mInfo.mGlobalMemCacheSize_byte   ), &mInfo.mGlobalMemCacheSize_byte   );
    GetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE         , sizeof(mInfo.mGlobalMemSize_byte        ), &mInfo.mGlobalMemSize_byte        );
    GetDeviceInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT      , sizeof(mInfo.mImage2DMaxHeight          ), &mInfo.mImage2DMaxHeight          );
    GetDeviceInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH       , sizeof(mInfo.mImage2DMaxWidth           ), &mInfo.mImage2DMaxWidth           );
    GetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH       , sizeof(mInfo.mImage3DMaxDepth           ), &mInfo.mImage3DMaxDepth           );
    GetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT      , sizeof(mInfo.mImage3DMaxHeight          ), &mInfo.mImage3DMaxHeight          );
    GetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH       , sizeof(mInfo.mImage3DMaxWidth           ), &mInfo.mImage3DMaxWidth           );
    GetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE          , sizeof(mInfo.mLocalMemSize_byte         ), &mInfo.mLocalMemSize_byte         );
    GetDeviceInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mInfo.mMaxConstantBufferSize_byte), &mInfo.mMaxConstantBufferSize_byte);
    GetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE      , sizeof(mInfo.mMaxMemAllocSize_byte      ), &mInfo.mMaxMemAllocSize_byte      );
    GetDeviceInfo(CL_DEVICE_MAX_PARAMETER_SIZE      , sizeof(mInfo.mMaxParameterSize_byte     ), &mInfo.mMaxParameterSize_byte     );
    GetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE     , sizeof(mInfo.mMaxWorkGroupSize          ), &mInfo.mMaxWorkGroupSize          );
    GetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES     , sizeof(mInfo.mMaxWorkItemSizes          ), &mInfo.mMaxWorkItemSizes          );

    GetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE       , sizeof(mInfo.mGlobalMemCacheType         ), &mInfo.mGlobalMemCacheType         );
    GetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE   , sizeof(mInfo.mGlobalMemCacheLineSize_byte), &mInfo.mGlobalMemCacheLineSize_byte);
    GetDeviceInfo(CL_DEVICE_LOCAL_MEM_TYPE              , sizeof(mInfo.mLocalMemType               ), &mInfo.mLocalMemType               );
    GetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS           , sizeof(mInfo.mMaxComputeUnits            ), &mInfo.mMaxComputeUnits            );
    GetDeviceInfo(CL_DEVICE_MAX_CONSTANT_ARGS           , sizeof(mInfo.mMaxConstantArgs            ), &mInfo.mMaxConstantArgs            );
    GetDeviceInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS         , sizeof(mInfo.mMaxReadImageArgs           ), &mInfo.mMaxReadImageArgs           );
    GetDeviceInfo(CL_DEVICE_MAX_SAMPLERS                , sizeof(mInfo.mMaxSamplers                ), &mInfo.mMaxSamplers                );
    GetDeviceInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS        , sizeof(mInfo.mMaxWriteImageArgs          ), &mInfo.mMaxWorkGroupSize           );
    GetDeviceInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN         , sizeof(mInfo.mMemBaseAddrAlign_bit       ), &mInfo.mMemBaseAddrAlign_bit       );
    GetDeviceInfo(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE    , sizeof(mInfo.mMinDataTypeAlignSize_byte  ), &mInfo.mMinDataTypeAlignSize_byte  );
    GetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR , sizeof(mInfo.mPreferredVectorWidthChar   ), &mInfo.mPreferredVectorWidthChar   );
    GetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(mInfo.mPreferredVectorWidthShort  ), &mInfo.mPreferredVectorWidthShort  );
    GetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT  , sizeof(mInfo.mPreferredVectorWidthInt    ), &mInfo.mPreferredVectorWidthInt    );
    GetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG , sizeof(mInfo.mPreferredVectorWidthLong   ), &mInfo.mPreferredVectorWidthLong   );
    GetDeviceInfo(CL_DEVICE_VENDOR_ID                   , sizeof(mInfo.mVendorId                   ), &mInfo.mVendorId                   );

    mInfo.mFlags.mAvailable         = GetDeviceInfo(CL_DEVICE_AVAILABLE         );
    mInfo.mFlags.mCompilerAvailable = GetDeviceInfo(CL_DEVICE_COMPILER_AVAILABLE);
    mInfo.mFlags.mEndianLittle      = GetDeviceInfo(CL_DEVICE_ENDIAN_LITTLE     );
    mInfo.mFlags.mImageSupport      = GetDeviceInfo(CL_DEVICE_IMAGE_SUPPORT     );

    GetDeviceInfo(CL_DRIVER_VERSION, sizeof(mInfo.mDriverVersion), &mInfo.mDriverVersion);
    GetDeviceInfo(CL_DEVICE_NAME   , sizeof(mInfo.mName         ), &mInfo.mName         );
    GetDeviceInfo(CL_DEVICE_PROFILE, sizeof(mInfo.mProfile      ), &mInfo.mProfile      );
    GetDeviceInfo(CL_DEVICE_VENDOR , sizeof(mInfo.mVendor       ), &mInfo.mVendor       );
    GetDeviceInfo(CL_DEVICE_VERSION, sizeof(mInfo.mVersion      ), &mInfo.mVersion      );
}

// ===== OpenCL =============================================================

// Exception  KmsLib::Exception *  See OCLW_GetDeviceInfo
// Threads  Apps
bool Processor_OpenCL::GetDeviceInfo(cl_device_info aParam)
{
    assert(0 != mDevice);

    cl_bool lResult   ;

    OCLW_GetDeviceInfo(mDevice, aParam, sizeof(lResult), &lResult);

    return lResult;
}

// Exception  KmsLib::Exception *  See OCLW_GetDeviceInfo
// Threads  Apps
void Processor_OpenCL::GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    assert(0 != mDevice);

    OCLW_GetDeviceInfo(mDevice, aParam, aOutSize_byte, aOut);
}

// Exception  KmsLib::Exception *  See OCLW_GetKernelWorkGroupInfo
// Threads  Apps
void Processor_OpenCL::GetKernelWorkGroupInfo(cl_kernel aKernel, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    assert(0 != mDevice);

    OCLW_GetKernelWorkGroupInfo(aKernel, mDevice, aParam, aOutSize_byte, aOut);
}
