
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread_Functions.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Function.h>

// ===== OpenNet ============================================================
#include "Kernel_Functions.h"
#include "Thread.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Functions : public Thread
{

public:

    Thread_Functions(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog);

    void AddAdapter(Adapter_Internal * aAdapter, const OpenNet::Function & aFunction);

    void AddDispatchCode();

    // ===== Thread =========================================================
    virtual void Prepare();

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);
    virtual void Processing_Wait (unsigned int aIndex);

    virtual void Release();

    virtual void Run_Loop ();
    virtual void Run_Start();

private:

    enum
    {
        EVENT_QTY = 3,
    };

    cl_event         mEvents[EVENT_QTY];
    Kernel_Functions mKernelFunctions;
    cl_program       mProgram        ;

};
