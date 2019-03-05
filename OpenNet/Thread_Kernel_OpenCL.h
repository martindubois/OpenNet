
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Kernel_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Thread_Kernel.h"
#include "Thread_OpenCL.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Kernel_OpenCL : public Thread_Kernel, public Thread_OpenCL
{

public:

    Thread_Kernel_OpenCL(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, cl_program aProgram, KmsLib::DebugLog * aDebugLog);

    void SetProgram(cl_program aProgram);

    // ===== Thread =========================================================

    virtual void Prepare();

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);

    virtual void Release();

};

