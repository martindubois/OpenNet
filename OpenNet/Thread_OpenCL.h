
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>

// ===== OpenNet ============================================================
#include "Processor_OpenCL.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_OpenCL
{

public:

    virtual ~Thread_OpenCL();

protected:

    Thread_OpenCL();

    void Prepare(Processor_OpenCL * aProcessor, Adapter_Vector * aAdapters, OpenNet::Kernel * aKernel, Buffer_Data_Vector * aBuffers);

    void Processing_Queue(const size_t * aGlobalSize, const size_t * aLocalSize, cl_event * aEvent);
    void Processing_Wait (cl_event aEvent, OpenNet::Kernel * aKernel);

    virtual void Release(OpenNet::Kernel * aKernel);

    cl_command_queue mCommandQueue;
    cl_kernel        mKernel_CL   ;
    cl_program       mProgram     ;

};