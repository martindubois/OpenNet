
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Thread_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <semaphore.h>

// ===== CUDA ===============================================================
#include <cuda.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>

// ===== OpenNet/CUDA =======================================================

#include "../Internal/Adapter_Internal.h"

#include "Processor_CUDA.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_CUDA
{

protected:

    Thread_CUDA( Processor_Internal * aProcessor, CUmodule aModule = NULL );

    virtual ~Thread_CUDA();

    void Prepare( Adapter_Vector * aAdapters, Buffer_Internal_Vector * aBuffers, bool aProfiling );

    void Processing_Queue(OpenNet::Kernel * aKernel, Event * aEvent, const size_t * aGlobalSize, const size_t * aLocalSize, void * * aArguments );

    void Release( OpenNet::Kernel * aKernel);

    void Run_Start();

    void         * * mArguments     ;
    CUfunction       mFunction      ;
    CUmodule         mModule        ;
    Processor_CUDA * mProcessor_CUDA;
    CUstream         mStream        ;

};
