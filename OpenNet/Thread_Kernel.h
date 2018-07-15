
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread_Kernel.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Thread.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Kernel : public Thread
{

public:

    Thread_Kernel(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, cl_program aProgram, KmsLib::DebugLog * aDebugLog);

    // ===== Thread =========================================================
    virtual ~Thread_Kernel();

protected:

    // ===== Thread =========================================================

    void Processing_Queue(unsigned int aIndex);
    void Processing_Wait (unsigned int aIndex);

    virtual void Run_Loop ();
    virtual void Run_Start();

};
