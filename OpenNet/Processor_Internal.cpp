
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNetK/Interface.h>
#include <OpenNetK/Types.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Processor_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aEnqueueMakeBufferResident [-K-;--X]
// aEnqueueWaitSignal         [-K-;--X]
// aDebugLog                  [-K-;RW-]
//
// Exception  KmsLib::Exception *  See InitInfo
//                                 See OCLW_CreateContext
//                                 See OCLW_CreateCommandQueueWithProperties
// Threads  Apps
Processor_Internal::Processor_Internal(cl_platform_id aPlatform, cl_device_id aDevice, clEnqueueMakeBuffersResidentAMD_fn aEnqueueMakeBufferResident, clEnqueueWaitSignalAMD_fn aEnqueueWaitSignal, KmsLib::DebugLog * aDebugLog)
    : mDebugLog                 (aDebugLog                 )
    , mDevice                   (aDevice                   )
    , mEnqueueMakeBufferResident(aEnqueueMakeBufferResident)
    , mEnqueueWaitSignal        (aEnqueueWaitSignal        )
{
    assert(   0 != aPlatform                 );
    assert(   0 != aDevice                   );
    assert(NULL != aEnqueueMakeBufferResident);
    assert(NULL != aEnqueueWaitSignal        );
    assert(NULL != aDebugLog                 );

    InitInfo();

    cl_context_properties lProperties[3];

    lProperties[0] = CL_CONTEXT_PLATFORM;
    lProperties[1] = (cl_context_properties)(aPlatform);
    lProperties[2] = 0;

    // OCLW_CreateContext ==> OCLW_ReleaseContext  See ~Processor_Internal
    mContext = OCLW_CreateContext(lProperties, 1, &aDevice);
    assert(NULL != mContext);

    // OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue  See ~Processor_Internal
    mQueue = OCLW_CreateCommandQueueWithProperties(mContext, aDevice);
    assert(NULL != mQueue);
}

// Exception  KmsLib::Exception *  See OCL_ReleaseCommandQueue
//                                 See OCL_ReleaseContext
// Threads  Apps
Processor_Internal::~Processor_Internal()
{
    assert(NULL != mContext);
    assert(NULL != mQueue  );

    // OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue  See Processor_Internal
    OCLW_ReleaseCommandQueue(mQueue);

    // OCLW_CreateContext ==> OCLW_ReleaseContext  See Processor_Internal
    OCLW_ReleaseContext(mContext);
}

// aFilterData [---;RW-]
// aBufferInfo [---;-W-]
// aBufferData [---;-W-]
//
// Exception  KmsLib::Exception *  CODE_OPEN_CL_ERROR
//                                 See GetKernelWorkGroupInfo
//                                 See OCLW_CreateBuffer
// Threads  Apps
//
// Buffer_Allocate ==> Buffer_Release
void Processor_Internal::Buffer_Allocate(unsigned int aPacketSize_byte, FilterData * aFilterData, OpenNet_BufferInfo * aBufferInfo, BufferData * aBufferData)
{
    assert(OPEN_NET_PACKET_SIZE_MAX_byte >= aPacketSize_byte    );
    assert(OPEN_NET_PACKET_SIZE_MIN_byte <= aPacketSize_byte    );
    assert(NULL                          != aFilterData         );
    assert(NULL                          != aFilterData->mKernel);
    assert(NULL                          != aBufferInfo         );
    assert(NULL                          != aBufferData         );

    assert(NULL != mContext);
    assert(NULL != mQueue  );

    memset(aBufferInfo, 0, sizeof(OpenNet_BufferInfo));
    memset(aBufferData, 0, sizeof(BufferData        ));

    size_t lPacketQty;

    GetKernelWorkGroupInfo(aFilterData->mKernel, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lPacketQty), &lPacketQty);

    aBufferInfo->mSize_byte = sizeof(OpenNet_BufferHeader);

    aBufferInfo->mSize_byte += sizeof(OpenNet_PacketInfo) * static_cast<unsigned int>(lPacketQty);
    aBufferInfo->mSize_byte += aPacketSize_byte           * static_cast<unsigned int>(lPacketQty);
    aBufferInfo->mSize_byte += (aBufferInfo->mSize_byte / (64 * 1024)) * aPacketSize_byte;

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See Buffer_Release
    aBufferData->mMem = OCLW_CreateBuffer(mContext, CL_MEM_BUS_ADDRESSABLE_AMD, aBufferInfo->mSize_byte);
    assert(NULL != aBufferData->mMem);

    cl_bus_address_amd lBusAddress;

    cl_int lStatus = mEnqueueMakeBufferResident(mQueue, 1, &aBufferData->mMem, CL_TRUE, &lBusAddress, 0, NULL, NULL);
    if (CL_SUCCESS != lStatus)
    {
        OCLW_ReleaseMemObject(aBufferData->mMem);

        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clEnqueueMakeBufferResident( , , , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    aBufferData->mPacketQty = static_cast<unsigned int>(lPacketQty);
    aBufferData->mSize_byte = aBufferInfo->mSize_byte;

    aBufferInfo->mBuffer_PA = lBusAddress.surface_bus_address;
    aBufferInfo->mMarker_PA = lBusAddress.marker_bus_address ;
    aBufferInfo->mPacketQty = static_cast<uint32_t>(lPacketQty);
}

// aBufferData [---;RW-]
//
// Exception  KmsLib::Exception *  CODE_INVALID_ARGUMENT
//                                 See OCLW_ReleaseMemObject
// Threads  Apps
//
// Buffer_Allocate ==> Buffer_Release
void Processor_Internal::Buffer_Release(BufferData * aBufferData)
{
    assert(NULL != aBufferData      );
    assert(NULL != aBufferData->mMem);

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See Buffer_Allocate
    OCLW_ReleaseMemObject(aBufferData->mMem);
}

// aFilterData [---;-W-]
// aFilter     [-K-;RW-]
//
// Threads  Apps
//
// Processing_Create ==> Processing_Release
void Processor_Internal::Processing_Create(FilterData * aFilterData, OpenNet::Filter * aFilter)
{
    assert(NULL != aFilterData);
    assert(NULL != aFilter    );

    memset(aFilterData, 0, sizeof(FilterData));

    aFilterData->mFilter = aFilter;

    // OCLW_CreateProgramWithSource ==> OCLW_ReleaseProgram  See Processing_Release
    aFilterData->mProgram = OCLW_CreateProgramWithSource(mContext, aFilter->GetCodeLineCount(), aFilter->GetCodeLines(), NULL);
    assert(NULL != aFilterData->mProgram);

    try
    {
        OCLW_BuildProgram(aFilterData->mProgram, 1, &mDevice, "-I V:/OpenNet/Includes", NULL, NULL);

        // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Processing_Release
        aFilterData->mKernel = OCLW_CreateKernel(aFilterData->mProgram, "Filter");
    }
    catch ( ... )
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        OCLW_GetProgramBuildInfo(aFilterData->mProgram, mDevice, CL_PROGRAM_BUILD_LOG, OpenNet::Filter::BUILD_LOG_MAX_SIZE_byte, aFilter->AllocateBuildLog());

        OCLW_ReleaseProgram(aFilterData->mProgram);
        aFilterData->mProgram = NULL;
        throw;
    }
}

// aFilterData [---;RW-]
// aBufferData [---;RW-]
//
// Processing_Queue ==> Processing_Wait
//
// Exception  KmsLib::Exception *  CODE_OPEN_CL_ERROR
//                                 See OCLW_EnqueueNDRangeKernel
//                                 See OCLW_SetKernelArg
// Thread  Worker
void Processor_Internal::Processing_Queue(FilterData * aFilterData, BufferData * aBufferData)
{
    assert(NULL != aFilterData            );
    assert(NULL != aFilterData->mFilter   );
    assert(NULL != aFilterData->mKernel   );
    assert(NULL != aBufferData            );
    assert(NULL != aBufferData->mMem      );
    assert(   0 <  aBufferData->mPacketQty);
    assert(   0 <  aBufferData->mSize_byte);

    size_t lGO = 0;
    size_t lGS = aBufferData->mPacketQty;

    OCLW_SetKernelArg(aFilterData->mKernel, 0, sizeof(aBufferData->mMem), &aBufferData->mMem);

    aFilterData->mFilter->AddKernelArgs(aFilterData->mKernel);

    cl_int lStatus = mEnqueueWaitSignal(mQueue, aBufferData->mMem, OPEN_NET_MARKER_VALUE, 0, NULL, NULL);
    if (CL_SUCCESS != lStatus)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "EnqueueWaitSignal( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    OCLW_EnqueueNDRangeKernel(mQueue, aFilterData->mKernel, 1, &lGO, &lGS, NULL, 0, NULL, &aBufferData->mEvent);
}

// aFilterData [---;RW-]
//
// Threads  Apps
//
// Processing_Create ==> Processing_Release
void Processor_Internal::Processing_Release(FilterData * aFilterData)
{
    assert(NULL != aFilterData          );
    assert(NULL != aFilterData->mKernel );
    assert(NULL != aFilterData->mProgram);

    // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Processing_Create
    OCLW_ReleaseKernel(aFilterData->mKernel);

    // OCLW_CreateProgramWithSournce ==> OCLW_ReleaseProgram  See Process_Create
    OCLW_ReleaseProgram(aFilterData->mProgram);
}

// aBufferData [---;RW-]
//
// Thread  Worker
//
// Processing_Queue ==> Processing_Wait
void Processor_Internal::Processing_Wait(BufferData * aBufferData)
{
    assert(NULL != aBufferData        );
    assert(NULL != aBufferData->mEvent);

    OCLW_WaitForEvents(1, &aBufferData->mEvent);
}

// ===== OpenNet::Processor =================================================

OpenNet::Status Processor_Internal::GetInfo(Info * aOut) const
{
    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(Info));

    return OpenNet::STATUS_OK;
}

const char * Processor_Internal::GetName() const
{
    return mInfo.mName;
}

OpenNet::Status Processor_Internal::Display(FILE * aOut) const
{
    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Info\n");

    return Processor::Display(mInfo, aOut);
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  See GetDeviceInfo
// Threads  Apps
void Processor_Internal::InitInfo()
{
    memset(&mInfo, 0, sizeof(mInfo));

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
bool Processor_Internal::GetDeviceInfo(cl_device_info aParam)
{
    assert(0 != mDevice);

    cl_bool lResult   ;

    OCLW_GetDeviceInfo(mDevice, aParam, sizeof(lResult), &lResult);

    return lResult;
}

// Exception  KmsLib::Exception *  See OCLW_GetDeviceInfo
// Threads  Apps
void Processor_Internal::GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    assert(0 != mDevice);

    OCLW_GetDeviceInfo(mDevice, aParam, aOutSize_byte, aOut);
}

// Exception  KmsLib::Exception *  See OCLW_GetKernelWorkGroupInfo
// Threads  Apps
void Processor_Internal::GetKernelWorkGroupInfo(cl_kernel aKernel, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    assert(0 != mDevice);

    OCLW_GetKernelWorkGroupInfo(aKernel, mDevice, aParam, aOutSize_byte, aOut);
}
