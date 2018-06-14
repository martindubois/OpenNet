
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Processor_Internal::Processor_Internal(cl_platform_id aPlatform, cl_device_id aDevice)
{
    assert(0 != aPlatform);
    assert(0 != aDevice  );

    mDevice = aDevice;

    InitInfo();

    cl_context_properties lProperties[3];

    lProperties[0] = CL_CONTEXT_PLATFORM;
    lProperties[1] = (cl_context_properties)(aPlatform);
    lProperties[2] = 0;

    cl_int lStatus;

    mContext = clCreateContext(lProperties, 1, &aDevice, NULL, NULL, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateContext( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    assert(0 != mContext);

    mQueue = clCreateCommandQueueWithProperties(mContext, aDevice, NULL, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateCommandQueue( , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    assert(0 != mQueue);
}

// ===== OpenNet::Processor =================================================

OpenNet::Status Processor_Internal::GetInfo(Info * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(Info));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Processor_Internal::Display(FILE * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Info\n");

    return Processor::Display(mInfo, aOut);
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Processor_Internal::InitInfo()
{
    assert(0 != mDevice);

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

bool Processor_Internal::GetDeviceInfo(cl_device_info aParam)
{
    assert(0 != mDevice);

    size_t  lInfo_byte;
    cl_bool lResult   ;

    cl_int lRet = clGetDeviceInfo(mDevice, aParam, sizeof(lResult), &lResult, &lInfo_byte);
    if (CL_SUCCESS != lRet)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet);
    }

    if (sizeof(lResult) < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo indicated an invalid size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }

    return lResult;
}

void Processor_Internal::GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    assert(0 != mDevice);

    size_t lInfo_byte;

    cl_int lRet = clGetDeviceInfo(mDevice, aParam, aOutSize_byte, aOut, &lInfo_byte);
    if (CL_SUCCESS != lRet)
    {
        char lMsg[1024];

        sprintf_s(lMsg, "clGetDeviceInfo( , , %llu bytes, ,  ) failed", aOutSize_byte);

        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo( , , , ,  ) failed", lMsg, __FILE__, __FUNCTION__, __LINE__, lRet);
    }

    if (aOutSize_byte < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo indicated an invalid size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }
}
