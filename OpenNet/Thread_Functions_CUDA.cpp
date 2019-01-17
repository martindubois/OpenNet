
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_CUDA.h

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "Thread_Functions_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Thread_Functions_CUDA::Thread_Functions_CUDA(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
{

}

// ===== Thread =============================================================

void Thread_Functions_CUDA::Prepare()
{
    assert(   0 < mAdapters.size());
    assert(NULL != mKernel        );
    assert(NULL != mProcessor     );

    assert(NULL == mCommandQueue);

    // Processor_Internal::CommandQueue_Create ==> OCLW_ReleaseCommandQueue  See Release
    mCommandQueue = mProcessor->CommandQueue_Create(mKernel->IsProfilingEnabled());
    assert(NULL != mCommandQueue);

    mKernel->SetCommandQueue(mCommandQueue);

    // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Release
    mKernel_CL = OCLW_CreateKernel(mProgram, "Filter");
    assert(NULL != mKernel_CL);

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        assert(NULL != mAdapters[i]);

        mAdapters[i]->Buffers_Allocate(mCommandQueue, mKernel_CL, &mBuffers);
    }
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

// CRITICAL PATH - Buffer
void Thread_Functions_CUDA::Processing_Queue(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(0 < mBuffers.size());
    assert(NULL != mBuffers[0]);

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
void Thread_Functions_CUDA::Processing_Wait(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(NULL != mEvents[aIndex]);

    Thread::Processing_Wait(mEvents[aIndex]);

    mEvents[aIndex] = NULL;
}

void Thread_Functions_CUDA::Run_Start()
{
    assert(0 < mBuffers.size());

    unsigned int i = 0;

    for (i = 0; i < mBuffers.size(); i++)
    {
        assert(NULL != mBuffers[i]->mMem);

        OCLW_SetKernelArg(mKernel_CL, i, sizeof(cl_mem), &mBuffers[i]->mMem);
    }

    Thread_Functions::Run_Start();
}
