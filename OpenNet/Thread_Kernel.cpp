
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread_Kernel.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Thread_Kernel.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aAdapter   [-K-;RW-]
// aKernel    [-K-;RW-]
// aProgram   [-K-;RW-]
// aDebugLog  [-K-;RW-]
Thread_Kernel::Thread_Kernel(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, cl_program aProgram, KmsLib::DebugLog * aDebugLog)
    : Thread(aProcessor, aDebugLog)
{
    assert(NULL != aProcessor);
    assert(NULL != aAdapter  );
    assert(NULL != aKernel   );
    assert(NULL != aProgram  );
    assert(NULL != aDebugLog );

    SetKernel (aKernel );
    SetProgram(aProgram);

    AddAdapter(aAdapter);
}

// ===== Thread =============================================================

Thread_Kernel::~Thread_Kernel()
{
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Kernel::Processing_Queue(unsigned int aIndex)
{
    assert(OPEN_NET_BUFFER_QTY > aIndex);

    assert(NULL != mCommandQueue);
    assert(NULL != mKernel_CL   );

    Buffer_Data * lBuffer = mBuffers[aIndex];
    assert(NULL != lBuffer        );
    assert(NULL == lBuffer->mEvent);
    assert(NULL != lBuffer->mMem  );

    OCLW_SetKernelArg(mKernel_CL, 0, sizeof(lBuffer->mMem), &lBuffer->mMem);

    mKernel->SetUserKernelArgs(mKernel_CL);

    // Here, we don't use event between the clEnqueueWaitSignal and the
    // clEnqueueNDRangeKernel because the command queue force the execution
    // order.
    OCLW_EnqueueWaitSignal(mCommandQueue, lBuffer->mMem, lBuffer->GetMarkerValue(), 0, NULL, NULL);

    size_t lGS = lBuffer->GetPacketQty();

    assert(0 < lGS);

    Thread::Processing_Queue(&lGS, NULL, &lBuffer->mEvent);
}

void Thread_Kernel::Processing_Wait(unsigned int aIndex)
{
    assert(OPEN_NET_BUFFER_QTY > aIndex);

    Buffer_Data * lBuffer = mBuffers[aIndex];
    assert(NULL != lBuffer        );
    assert(NULL != lBuffer->mEvent);

    Thread::Processing_Wait(lBuffer->mEvent);

    lBuffer->mEvent = NULL;
}

void Thread_Kernel::Run_Loop()
{
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
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }
}

void Thread_Kernel::Run_Start()
{
    assert(                  0 <  mBuffers.size());
    assert(OPEN_NET_BUFFER_QTY >= mBuffers.size());

    for (unsigned int i = 0; i < mBuffers.size(); i++)
    {
        Processing_Queue(i);
    }
}
