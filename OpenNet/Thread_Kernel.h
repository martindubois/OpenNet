
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Kernel.h

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

    Thread_Kernel(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, KmsLib::DebugLog * aDebugLog);

    // ===== Thread =========================================================
    virtual ~Thread_Kernel();

    virtual void Processing_Wait( unsigned int aIndex );

};
