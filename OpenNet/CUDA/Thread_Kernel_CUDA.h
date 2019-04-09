
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Thread_Kernel_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet/CUDA =======================================================

#include "../Thread_Kernel.h"

#include "Thread_CUDA.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_Kernel_CUDA : public Thread_Kernel, public Thread_CUDA
{

public:

    Thread_Kernel_CUDA(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, CUmodule aModule, KmsLib::DebugLog * aDebugLog);

    // ===== Thread =========================================================

    virtual void Prepare();

protected:

    // ===== Thread =========================================================

    virtual void Processing_Queue(unsigned int aIndex);

    virtual void Run_Start();

    virtual void Release();

};
