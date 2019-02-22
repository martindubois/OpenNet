
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <semaphore.h>

// ===== CUDA ===============================================================
#include <cuda.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

#include "Processor_CUDA.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread_CUDA
{

public:

// internal

    void KernelCompleted();

protected:

    Thread_CUDA();
    Thread_CUDA( CUmodule aModule );

    virtual ~Thread_CUDA();

    void Prepare( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers );
    void Prepare( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers, unsigned int aQueueDepth );

    void Processing_Queue(OpenNet::Kernel * aKernel, const size_t * aGlobalSize, const size_t * aLocalSize, void * * aArguments );
    void Processing_Wait ();

    void Release(OpenNet::Kernel * aKernel);

    void Run_Start(Processor_Internal * aProcessor);

    void   * * mArguments;
    CUfunction mFunction ;
    CUmodule   mModule   ;
    CUstream   mStream   ;

private:

    void Prepare_Internal( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers );

    sem_t mSemaphore;

};
