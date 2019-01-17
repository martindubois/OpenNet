
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions.h

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
#include "Thread_Functions.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aProfilingEnabled
// aDebugLog  [-K-;RW-]
Thread_Functions::Thread_Functions(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
    : Thread(aProcessor, aDebugLog)
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );

    if (aProfilingEnabled)
    {
        mKernelFunctions.EnableProfiling();
    }
    else
    {
        mKernelFunctions.DisableProfiling();
    }

    SetKernel(&mKernelFunctions);
}

// aAdapter  [-K-;RW-]
// aFunction [---;R--]
void Thread_Functions::AddAdapter(Adapter_Internal * aAdapter, const OpenNet::Function & aFunction)
{
    assert(NULL !=   aAdapter  );
    assert(NULL != (&aFunction));

    mKernelFunctions.AddFunction(aFunction);

    Thread::AddAdapter(aAdapter);
}

void Thread_Functions::AddDispatchCode()
{
    unsigned int lAdapterCount = static_cast<unsigned int>(mAdapters.size());
    assert(0 < lAdapterCount);

    unsigned int * lBufferQty = new unsigned int[lAdapterCount];
    assert(NULL != lBufferQty);

    for (unsigned int i = 0; i < lAdapterCount; i++)
    {
        assert(NULL != mAdapters[i]);

        lBufferQty[i] = mAdapters[i]->GetBufferQty();
    }

    mKernelFunctions.AddDispatchCode(lBufferQty);

    delete[] lBufferQty;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Functions::Release()
{
    assert(NULL != mProcessor);

    mProcessor->Thread_Release();
}

// CRITICAL PATH
void Thread_Functions::Run_Loop()
{
    assert(NULL != mDebugLog);

    try
    {
        unsigned lIndex = 0;

        while (IsRunning())
        {
            Run_Iteration(lIndex);

            lIndex = (lIndex + 1) % EVENT_QTY;
        }

        for (unsigned int i = 0; i < EVENT_QTY; i++)
        {
            Processing_Wait(lIndex);

            lIndex = (lIndex + 1) % EVENT_QTY;
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }
}

void Thread_Functions::Run_Start()
{
    for (unsigned int i = 0; i < EVENT_QTY; i++)
    {
        Processing_Queue(i);
    }
}
