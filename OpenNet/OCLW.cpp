
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/OCLW.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

// TODO  Test
//
// OCLW_CreateContext ==> OCLW_ReleaseContext
cl_context OCLW_CreateContext(const cl_context_properties * aProperties, cl_uint aDeviceCount, const cl_device_id * aDevices)
{
    assert(NULL != aProperties );
    assert(   0 <  aDeviceCount);
    assert(NULL != aDevices    );

    cl_int lStatus;

    // clCreateContext ==> clReleaseContext  See OCLW_ReleaseContext
    cl_context lResult = clCreateContext(aProperties, aDeviceCount, aDevices, NULL, NULL, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateContext( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    return lResult;
}

// NOT TESTED  OpenNet.OCLW
//             clGetPlatformIDs fail
void OCLW_GetPlatformIDs(cl_uint aCount, cl_platform_id * aOut, cl_uint * aInfo)
{
    cl_int lStatus = clGetPlatformIDs(aCount, aOut, aInfo);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetPlatformIDs( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
void OCLW_WaitForEvents(cl_uint aCount, const cl_event * aEvents)
{
    assert(0    <  aCount );
    assert(NULL != aEvents);

    cl_int lStatus = clWaitForEvents(aCount, aEvents);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clWaitForEvents( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_command_queue ===================================================

// TODO  Test
void OCLW_EnqueueNDRangeKernel(cl_command_queue aCommandQueue, cl_kernel aKernel, cl_uint aWorkDim, const size_t * aGlobalWorkOffset, const size_t * aGlobalWorkSize, const size_t * aLocalWorkSize, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent)
{
    assert(NULL != aCommandQueue    );
    assert(NULL != aKernel          );
    assert(   1 <= aWorkDim         );
    assert(NULL != aGlobalWorkOffset);
    assert(NULL != aGlobalWorkSize  );
    assert(NULL != aEvent           );

    cl_int lStatus = clEnqueueNDRangeKernel(aCommandQueue, aKernel, aWorkDim, aGlobalWorkOffset, aGlobalWorkSize, aLocalWorkSize, aEventCount, aEvents, aEvent);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clEnqueueNDRangeKernel( , , , , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
void OCLW_EnqueueReadBuffer(cl_command_queue aCommandQueue, cl_mem aBuffer, cl_bool aBlockingRead, size_t aOffset_byte, size_t aSize_byte, void * aOut, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent)
{
    assert(NULL != aCommandQueue);
    assert(NULL != aBuffer      );
    assert(   0 <  aSize_byte   );
    assert(NULL != aOut         );

    cl_int lStatus = clEnqueueReadBuffer(aCommandQueue, aBuffer, aBlockingRead, aOffset_byte, aSize_byte, aOut, aEventCount, aEvents, aEvent);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clEnqueueReadBuffer( , , , , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
//
// OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue
void OCLW_ReleaseCommandQueue(cl_command_queue aCommandQueue)
{
    assert(NULL != aCommandQueue);

    // clCreateCommandQueueWithProperties ==> clReleaseCommandQueue  See OCLW_CreateCommandQueueWithProperties
    cl_int lStatus = clReleaseCommandQueue(aCommandQueue);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseCommandQueue(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_context =========================================================

// TODO  Test
//
// OCLE_CreateBuffer ==> OCLW_ReleaseMemObject
cl_mem OCLW_CreateBuffer(cl_context aContext, cl_mem_flags aFlags, size_t aSize_byte)
{
    assert(NULL != aContext  );
    assert(   0 <  aSize_byte);

    cl_int lStatus;

    // clCreateBuffer ==> clReleaseMemObject  See OCLW_ReleaseMemObject
    cl_mem lResult = clCreateBuffer(aContext, aFlags, aSize_byte, NULL, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateBuffer( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    return lResult;
}

// TODO  Test
//
// OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue
cl_command_queue OCLW_CreateCommandQueueWithProperties(cl_context aContext, cl_device_id aDevice, const cl_queue_properties * aProperties)
{
    assert(NULL != aContext);
    assert(NULL != aDevice );

    cl_int lStatus;

    // clCreateCommandQueueWithProperties ==> clReleaseCommandQueue  See OCLW_ReleaseCommandQueue
    cl_command_queue lResult = clCreateCommandQueueWithProperties(aContext, aDevice, aProperties, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateCommandQueueWithProperties( , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    return lResult;
}

// TODO  Test
//
// OCLW_CreateProgramWithSource ==> OCLW_ReleaseProgram
cl_program OCLW_CreateProgramWithSource(cl_context aContext, cl_uint aLineCount, const char ** aLines, const size_t * aLineLengths)
{
    assert(NULL != aContext    );
    assert(   0 <  aLineCount  );
    assert(NULL != aLines      );
    assert(NULL != aLineLengths);

    cl_int lStatus;

    // clCreateProgramWithSource ==> clReleaseProgram  See OCLW_ReleaseProgram
    cl_program lResult = clCreateProgramWithSource(aContext, aLineCount, aLines, aLineLengths, &lStatus);

    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateProgramWithSource( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    assert(NULL != lResult);

    return lResult;
}

// TODO  Test
//
// OCLW_CreateContext ==> OCLW_ReleaseContext
void OCLW_ReleaseContext(cl_context aContext)
{
    assert(NULL != aContext);

    // clCreateContext ==> clReleaseContext  See OCLW_CreateContext
    cl_int lStatus = clReleaseContext(aContext);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseContext(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_device_id =======================================================

// TODO  Test
void OCLW_GetDeviceInfo(cl_device_id aDevice, cl_device_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(NULL != aDevice      );
    assert(   0 <  aOutSize_byte);
    assert(NULL != aOut         );

    size_t lInfo_byte;

    cl_int lStatus = clGetDeviceInfo(aDevice, aParam, aOutSize_byte, aOut, &lInfo_byte);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    if (aOutSize_byte < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceInfo reported an invalid data size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }
}

// ===== cl_event ===========================================================

void OCLW_ReleaseEvent(cl_event aEvent)
{
    assert(NULL != aEvent);

    cl_int lStatus = clReleaseEvent(aEvent);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseEvent(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_kernel ==========================================================

// TODO  Test
void OCLW_GetKernelWorkGroupInfo(cl_kernel aKernel, cl_device_id aDevice, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(NULL != aKernel      );
    assert(NULL != aDevice      );
    assert(   0 <  aOutSize_byte);
    assert(NULL != aOut         );

    size_t lInfo_byte;

    cl_int lStatus = clGetKernelWorkGroupInfo(aKernel, aDevice, aParam, aOutSize_byte, aOut, &lInfo_byte);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetKernelWorkGroupInfo( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    if (aOutSize_byte < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetKernelWorkGroupInfo reported an invalid data size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }
}

// TODO  Test
//
// OCLW_CreateKernel ==> OCLW_ReleaseKernel
void OCLW_ReleaseKernel(cl_kernel aKernel)
{
    assert(NULL != aKernel);

    // clCreateKernel ==> clReleaseKernel  See OCLW_CreateKernel
    cl_int lStatus = clReleaseKernel(aKernel);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseKernel(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
void OCLW_SetKernelArg(cl_kernel aKernel, cl_uint aIndex, size_t aSize_byte, cl_mem * aMemObject)
{
    assert(NULL != aKernel   );
    assert(   0 <  aSize_byte);
    assert(NULL != aMemObject);

    cl_int lStatus = clSetKernelArg(aKernel, aIndex, aSize_byte, aMemObject);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clSetKernelArg( , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_mem =============================================================

// TODO  Test
//
// OCLW_CreateBuffer ==> OCLW_ReleaseMemObject
void OCLW_ReleaseMemObject(cl_mem aMemObject)
{
    assert(NULL != aMemObject);

    // clCreateBuffer ==> clReleaseMemObject  See OCLW_CreateBuffer
    cl_int lStatus = clReleaseMemObject(aMemObject);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseMemObject(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// ===== cl_platform_id =====================================================

// TODO  Test
void OCLW_GetDeviceIDs(cl_platform_id aPlatform, cl_device_type aType, cl_uint aCount, cl_device_id * aOut, cl_uint * aInfo)
{
    assert(   0 != aPlatform);
    assert(   0 <  aCount   );
    assert(NULL != aOut     );
    assert(NULL != aInfo    );

    cl_int lStatus = clGetDeviceIDs(aPlatform, aType, aCount, aOut, aInfo);
    switch (lStatus)
    {
    case CL_SUCCESS: break;

    case CL_DEVICE_NOT_FOUND:
        (*aInfo) = 0;
        break;

    default :
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetDeviceIDs( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
void * OCLW_GetExtensionFunctionAddressForPlatform(cl_platform_id aPlatform, const char * aName)
{
    void * lResult = clGetExtensionFunctionAddressForPlatform(aPlatform, aName);
    if (NULL == lResult)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetExtensionFunctionAddressForPlatform( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    return lResult;
}

// TODO  Test
void OCLW_GetPlatformInfo(cl_platform_id aPlatform, cl_platform_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(0    != aPlatform    );
    assert(0    <  aOutSize_byte);
    assert(NULL != aOut         );

    size_t lInfo_byte;

    cl_int lStatus = clGetPlatformInfo(aPlatform, aParam, aOutSize_byte, aOut, &lInfo_byte);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetPlatformInfo( , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    if (aOutSize_byte < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetPlatformInfo repported and invalid data size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }
}

// ===== cl_program =========================================================

// TODO  Test
void OCLW_BuildProgram(cl_program aProgram, cl_uint aDeviceCount, const cl_device_id * aDevices, const char * aOptions, void (*aNotify)(cl_program, void *), void * aUserData)
{
    assert(NULL != aProgram    );
    assert(   1 <= aDeviceCount);
    assert(NULL != aDevices    );
    assert(NULL != aOptions    );

    cl_int lStatus = clBuildProgram(aProgram, aDeviceCount, aDevices, aOptions, aNotify, aUserData);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clBuildProgram( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// TODO  Test
//
// OCLW_CreateKernel ==> OCLW_ReleaseKernel
cl_kernel OCLW_CreateKernel(cl_program aProgram, const char * aName)
{
    assert(NULL != aProgram);
    assert(NULL != aName   );

    cl_int lStatus;

    // clCreateKernel ==> clReleaseKernel  OCLW_ReleaseKernel
    cl_kernel lResult = clCreateKernel(aProgram, aName, &lStatus);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clCreateKernel( , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    return lResult;
}

void OCLW_GetProgramBuildInfo(cl_program aProgram, cl_device_id aDevice, cl_program_build_info aParam, size_t aOutSize_byte, void * aOut)
{
    assert(NULL != aProgram     );
    assert(   0 <  aOutSize_byte);
    assert(NULL != aOut         );

    size_t lInfo_byte;

    cl_int lStatus = clGetProgramBuildInfo(aProgram, aDevice, aParam, aOutSize_byte, aOut, &lInfo_byte);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetProgramBuildInfo( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    if (aOutSize_byte < lInfo_byte)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clGetProgramBuildInfo reported an invalid data size", NULL, __FILE__, __FUNCTION__, __LINE__, static_cast<unsigned int>(lInfo_byte));
    }
}

// TODO  Test
//
// OCLW_CreateProgramWithSource ==> OCLW_ReleaseProgram
void OCLW_ReleaseProgram(cl_program aProgram)
{
    assert(NULL != aProgram);

    // clCreateProgramWithSouce ==> clReleaseProgram  See OCLW_CreateProgramWithSource
    cl_int lStatus = clReleaseProgram(aProgram);
    if (CL_SUCCESS != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_OPEN_CL_ERROR,
            "clReleaseProgram(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}
