
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/OCLW.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern cl_context OCLW_CreateContext (const cl_context_properties * aProperties, cl_uint aDeviceCount, const cl_device_id * aDevices);
extern void       OCLW_GetPlatformIDs(cl_uint aCount, cl_platform_id * aOut, cl_uint * aInfo);
extern void       OCLW_WaitForEvents (cl_uint aCount, const cl_event * aEvents);

// ===== cl_command_queue ===================================================
extern void OCLW_EnqueueMakeBufferResident(cl_command_queue aCommandQueue, cl_uint aMemCount, cl_mem * aMems, cl_bool aBlocking, cl_bus_address_amd * aAddress, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
extern void OCLW_EnqueueNDRangeKernel     (cl_command_queue aCommandQueue, cl_kernel aKernel, cl_uint aWorkDim, const size_t * aGlobalWorkOffset, const size_t * aGlobalWorkSize, const size_t * aLocalWorkSize, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
extern void OCLW_EnqueueReadBuffer        (cl_command_queue aCommandQueue, cl_mem aBuffer, cl_bool aBlocking, size_t aOffset_byte, size_t aSize_byte,      void * aOut, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
extern void OCLW_EnqueueWaitSignal        (cl_command_queue aCommandQueue, cl_mem aBuffer, int aValue, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
extern void OCLW_EnqueueWriteBuffer       (cl_command_queue aCommandQueue, cl_mem aBuffer, cl_bool aBlocking, size_t aOffset_byte, size_t aSize_byte, const void * aIn, cl_uint aEventCount, const cl_event * aEvents, cl_event * aEvent);
extern void OCLW_Flush                    (cl_command_queue aCommandQueue);
extern void OCLW_ReleaseCommandQueue      (cl_command_queue aCommandQueue);

// ===== cl_context =========================================================
extern cl_mem           OCLW_CreateBuffer                    (cl_context aContext, cl_mem_flags aFlags, size_t aSize_byte);
extern cl_command_queue OCLW_CreateCommandQueueWithProperties(cl_context aContext, cl_device_id aDevice, const cl_queue_properties * aProperties);
extern cl_program       OCLW_CreateProgramWithSource         (cl_context aContext, cl_uint aLineCount, const char ** aLines, const size_t * aLineLengths);
extern void             OCLW_ReleaseContext                  (cl_context aContext);

// ===== cl_device_id =======================================================
extern void OCLW_GetDeviceInfo(cl_device_id aDevice, cl_device_info aParam, size_t aOutSize_byte, void * aOut);

// ===== cl_event ===========================================================
extern void OCLW_GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam, size_t aOutSize_byte, void * aOut);
extern void OCLW_ReleaseEvent         (cl_event aEvent);

// ===== cl_kernel ==========================================================
extern void OCLW_GetKernelWorkGroupInfo(cl_kernel aKernel, cl_device_id aDevice, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut);
extern void OCLW_ReleaseKernel         (cl_kernel aKernel);
extern void OCLW_SetKernelArg          (cl_kernel aKernel, cl_uint aIndex, size_t aSize_byte, cl_mem * aMemObject);

// ===== cl_mem =============================================================
extern void OCLW_ReleaseMemObject(cl_mem aMemObject);

// ===== cl_platform_id =====================================================
extern void   OCLW_GetDeviceIDs                          (cl_platform_id aPlatform, cl_device_type aType, cl_uint aCount, cl_device_id * aOut, cl_uint * aInfo);
extern void * OCLW_GetExtensionFunctionAddressForPlatform(cl_platform_id aPlatform, const char * aName);
extern void   OCLW_GetPlatformInfo                       (cl_platform_id aPlatform, cl_platform_info aParam, size_t aOutSize_byte, void * aOut);
extern void   OCLW_Initialise                            (cl_platform_id aPlatform);

// ===== cl_program =========================================================
extern void      OCLW_BuildProgram       (cl_program aProgram, cl_uint aDeviceCount, const cl_device_id * aDevices, const char * aOptions, void(*aNotify)(cl_program, void *), void * aUserData);
extern cl_kernel OCLW_CreateKernel       (cl_program aProgram, const char * aName);
extern void      OCLW_GetProgramBuildInfo(cl_program aProgram, cl_device_id aDevice, cl_program_build_info aParam, size_t aOutSize_byte, void * aOut);
extern void      OCLW_ReleaseProgram     (cl_program aProgram);
