
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Thread_CUDA.h"
#include "Thread_Functions.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Functions_CUDA : public Thread_Functions, public Thread_CUDA
{

public:

    Thread_Functions_CUDA(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog);

    // ===== Thread =========================================================

    virtual void Prepare();

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);

    virtual void Run_Start();

    virtual void Release();

private:

    Event_CUDA mEvent_CUDA[ QUEUE_DEPTH ];

};
