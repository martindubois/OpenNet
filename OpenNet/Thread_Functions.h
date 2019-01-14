
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions.h

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

protected:

    enum
    {
        EVENT_QTY = 3,
    };

    // ===== Thread =========================================================

    virtual void Release();

    virtual void Run_Loop ();

    Kernel_Functions mKernelFunctions;

};
