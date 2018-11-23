
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread_Functions.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Thread_Functions.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
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
    assert(NULL == mProgram);

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

// ===== Thread =============================================================

void Thread_Functions::Prepare()
{
    mProgram = mProcessor->Program_Create(&mKernelFunctions);
    assert(NULL != mProgram);

    SetProgram(mProgram);

    Thread::Prepare();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

// CRITICAL PATH - Buffer
void Thread_Functions::Processing_Queue(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(0    <  mBuffers.size());
    assert(NULL != mBuffers[0]    );

    size_t lLS = mBuffers[0]->GetPacketQty();
    size_t lGS = lLS * mBuffers.size();

    assert(0 < lLS);

    Thread::Processing_Queue(&lGS, &lLS, mEvents + aIndex);
}

// CRITICAL_PATH
//
// Thread  Worker
//
// Processing_Queue ==> Processing_Wait
void Thread_Functions::Processing_Wait(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(NULL != mEvents[aIndex]);

    Thread::Processing_Wait(mEvents[aIndex]);

    mEvents[aIndex] = NULL;
}

void Thread_Functions::Release()
{
    assert(NULL != mProcessor);

    mProcessor->Thread_Release();

    Thread::Release();
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
    assert(0 < mBuffers.size());

    unsigned int i = 0;

    for (i = 0; i < mBuffers.size(); i++)
    {
        assert(NULL != mBuffers[i]->mMem);

        OCLW_SetKernelArg(mKernel_CL, i, sizeof(cl_mem), &mBuffers[i]->mMem);
    }

    for (i = 0; i < EVENT_QTY; i++)
    {
        Processing_Queue(i);
    }
}
