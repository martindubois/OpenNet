
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

    // ===== Thread =========================================================

    virtual void Prepare();

protected:

    enum
    {
        QUEUE_DEPTH = 3,
    };

    // ===== Thread =========================================================

    virtual void Processing_Wait( unsigned int aIndex );

    virtual void Release();

    Event          * mEvents[ QUEUE_DEPTH ];
    Kernel_Functions mKernelFunctions;

};
