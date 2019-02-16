
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

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Kernel::Run_Loop()
{
    // printf( __CLASS__ "Run_Loop()\n" );

    assert(                  0 <  mBuffers.size());
    assert(OPEN_NET_BUFFER_QTY >= mBuffers.size());
    assert(NULL                != mDebugLog      );

    try
    {
        unsigned lIndex = 0;

        while (IsRunning())
        {
            Run_Iteration(lIndex);

            lIndex = (lIndex + 1) % mBuffers.size();
        }

        for (unsigned int i = 0; i < mBuffers.size(); i++)
        {
            Processing_Wait(lIndex);

            lIndex = (lIndex + 1) % mBuffers.size();
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Run_Loop", __LINE__);
        mDebugLog->Log(eE);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Run_Loop", __LINE__);
    }
}

void Thread_Kernel::Run_Start()
{
    // printf( __CLASS__ "Run_Start()\n" );

    assert(                  0 <  mBuffers.size());
    assert(OPEN_NET_BUFFER_QTY >= mBuffers.size());

    for (unsigned int i = 0; i < mBuffers.size(); i++)
    {
        Processing_Queue(i);
    }
}
