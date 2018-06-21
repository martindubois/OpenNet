
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/OCLW.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

cl_context OCLW_CreateContext (const cl_context_properties * aProperties, cl_uint aDeviceCount, const cl_device_id * aDevices);
void       OCLW_GetPlatformIDs(cl_uint aCount, cl_platform_id * aOut, cl_uint * aInfo);
void       OCLW_WaitForEvents (cl_uint aCount, const cl_event * aEvents);

// ===== cl_command_queue ===================================================
void OCLW_EnqueueNDRangeKernel(cl_command_queue aCommandQueue, cl_kernel aKernel, cl_uint aWorkDim, const size_t * aGlobalWorkOffset, const size_t * aGlobalWorkSize, const size_t * aLocalWorkSize, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
void OCLW_EnqueueReadBuffer   (cl_command_queue aCommandQueue, cl_mem aBuffer, cl_bool aBlockingRead, size_t aOffset_byte, size_t aSize_byte, void * aOut, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
void OCLW_ReleaseCommandQueue (cl_command_queue aCommandQueue);

// ===== cl_context =========================================================
cl_mem           OCLW_CreateBuffer                    (cl_context aContext, cl_mem_flags aFlags, size_t aSize_byte);
cl_command_queue OCLW_CreateCommandQueueWithProperties(cl_context aContext, cl_device_id aDevice);
cl_program       OCLW_CreateProgramWithSource         (cl_context aContext, cl_uint aLineCount, const char ** aLines, const size_t * aLineLengths);
void             OCLW_ReleaseContext                  (cl_context aContext);

// ===== cl_device_id =======================================================
void OCLW_GetDeviceInfo(cl_device_id aDevice, cl_device_info aParam, size_t aOutSize_byte, void * aOut);

// ===== cl_kernel ==========================================================
void OCLW_GetKernelWorkGroupInfo(cl_kernel aKernel, cl_device_id aDevice, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut);
void OCLW_ReleaseKernel         (cl_kernel aKernel);
void OCLW_SetKernelArg          (cl_kernel aKernel, cl_uint aIndex, size_t aSize_byte, cl_mem * aMemObject);

// ===== cl_mem =============================================================
void OCLW_ReleaseMemObject(cl_mem aMemObject);

// ===== cl_platform_id =====================================================
void   OCLW_GetDeviceIDs                          (cl_platform_id aPlatform, cl_device_type aType, cl_uint aCount, cl_device_id * aOut, cl_uint * aInfo);
void * OCLW_GetExtensionFunctionAddressForPlatform(cl_platform_id aPlatform, const char * aName);
void   OCLW_GetPlatformInfo                       (cl_platform_id aPlatform, cl_platform_info aParam, size_t aOutSize_byte, void * aOut);

// ===== cl_program =========================================================
void      OCLW_BuildProgram       (cl_program aProgram, cl_uint aDeviceCount, const cl_device_id * aDevices, const char * aOptions, void(*aNotify)(cl_program, void *), void * aUserData);
cl_kernel OCLW_CreateKernel       (cl_program aProgram, const char * aName);
void      OCLW_GetProgramBuildInfo(cl_program aProgram, cl_device_id aDevice, cl_program_build_info aParam, size_t aOutSize_byte, void * aOut);
void      OCLW_ReleaseProgram     (cl_program aProgram);
