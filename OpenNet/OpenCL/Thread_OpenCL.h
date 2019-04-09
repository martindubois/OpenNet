
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

class Event_OpenCL;

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_OpenCL
{

public:

    virtual ~Thread_OpenCL();

protected:

    Thread_OpenCL();

    void Prepare(Processor_OpenCL * aProcessor, Adapter_Vector * aAdapters, OpenNet::Kernel * aKernel, Buffer_Internal_Vector * aBuffers);

    void Processing_Queue(Event_OpenCL * aEvent, const size_t * aGlobalSize, const size_t * aLocalSize);

    void Release(OpenNet::Kernel * aKernel);

    cl_command_queue mCommandQueue;
    cl_kernel        mKernel_CL   ;
    cl_program       mProgram     ;

};
