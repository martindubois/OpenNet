
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Windows.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Windows : public Adapter_Internal
{

public:

    Adapter_Windows(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    void Buffers_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Data_Vector * aBuffers);

    // ===== Adapter_Internal ===============================================
    
    virtual void Connect(IoCtl_Connect_In * aConnect);

    // ===== OpenNet::Adapter ===============================================

    virtual ~Adapter_Windows();

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

protected:

    // ===== Adapter_Internal ===============================================

    virtual void ResetInputFilter_Internal();
    virtual void SetInputFilter_Internal  (OpenNet::Kernel * aKernel);

    virtual Thread * Thread_Prepare_Internal(OpenNet::Kernel * aKernel);

private:

    Buffer_Data * Buffer_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel);

    cl_program mProgram;

};
