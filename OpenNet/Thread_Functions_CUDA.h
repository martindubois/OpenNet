
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Thread_Functions.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Functions_CUDA : public Thread_Functions
{

public:

    Thread_Functions_CUDA(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog);

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);
    virtual void Processing_Wait (unsigned int aIndex);

    virtual void Prepare_Internal();

    virtual void Run_Start();

};
