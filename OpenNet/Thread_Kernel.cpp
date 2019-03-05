
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Kernel.coo

#define __CLASS__ "Thread_Kernel::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== OpenNet ============================================================
#include "Thread_Kernel.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aAdapter   [-K-;RW-]
// aKernel    [-K-;RW-]
// aDebugLog  [-K-;RW-]
Thread_Kernel::Thread_Kernel(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, KmsLib::DebugLog * aDebugLog)
    : Thread(aProcessor, aDebugLog)
{
    assert(NULL != aProcessor);
    assert(NULL != aAdapter  );
    assert(NULL != aKernel   );
    assert(NULL != aDebugLog );

    SetKernel (aKernel );

    AddAdapter(aAdapter);
}

// ===== Thread =============================================================

Thread_Kernel::~Thread_Kernel()
{
}

void Thread_Kernel::Processing_Wait(unsigned int aIndex)
{
    assert( NULL != mBuffers[ aIndex ] );
    assert( NULL != mKernel            );

    Thread::Processing_Wait( mKernel, mBuffers[ aIndex ]->GetEvent() );
}
