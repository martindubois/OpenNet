
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Thread_Functions.h"
#include "Thread_OpenCL.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Functions_OpenCL : public Thread_Functions, public Thread_OpenCL
{

public:

    Thread_Functions_OpenCL(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog);

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);
    virtual void Processing_Wait (unsigned int aIndex);

    virtual void Prepare_Internal();

    virtual void Release();

    virtual void Run_Start();

private:

    cl_command_queue mCommandQueue;
    cl_event         mEvents[EVENT_QTY];
    cl_kernel        mKernel_CL   ;
    cl_program       mProgram     ;

};
