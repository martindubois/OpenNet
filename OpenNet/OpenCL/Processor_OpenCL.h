
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/OpenCL/Processor_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet/OpenCL =====================================================
#include "../Internal/Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_OpenCL : public Processor_Internal
{

public:

    Processor_OpenCL(cl_platform_id aPlatform, cl_device_id aDevice, KmsLib::DebugLog * aDebugLog);

    Buffer_Internal * Buffer_Allocate(bool aProfiling, unsigned int aPacketSize_byte, cl_command_queue aCommandQueue, cl_kernel aKernel, OpenNetK::Buffer * aBuffer);

    cl_command_queue CommandQueue_Create(bool aProfilingEnabled);

    cl_program Program_Create(OpenNet::Kernel * aKernel, unsigned int aAdapterNo);

    // ====== Processor_Internal ============================================

    virtual OpenNet::UserBuffer * AllocateUserBuffer_Internal(unsigned int aSize_byte);

    virtual Thread_Functions * Thread_Get();

    // ===== OpenNet::Processor =============================================

    virtual ~Processor_OpenCL();

    virtual void          * GetContext();
    virtual void          * GetDevice ();

private:

    void InitInfo();

    // ===== OpenCL =========================================================

    bool GetDeviceInfo(cl_device_info aParam);
    void GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut);

    void GetKernelWorkGroupInfo(cl_kernel aKernel, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut);

    cl_context   mContext;
    cl_device_id mDevice ;

};
