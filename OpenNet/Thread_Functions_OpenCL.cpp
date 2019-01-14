
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_OpenCL.h

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet ============================================================
#include "Adapter_Windows.h"
#include "Buffer_Data_OpenCL.h"
#include "OCLW.h"
#include "Processor_OpenCL.h"

#include "Thread_Functions_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aProfilingEnabled
// aDebugLog  [-K-;RW-]
Thread_Functions_OpenCL::Thread_Functions_OpenCL(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
    : Thread_Functions(aProcessor, aProfilingEnabled, aDebugLog)
    , mCommandQueue(NULL)
    , mKernel_CL   (NULL)
    , mProgram     (NULL)
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

// CRITICAL PATH - Buffer
void Thread_Functions_OpenCL::Processing_Queue(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(0 < mBuffers.size());
    assert(NULL != mBuffers[0]);

    size_t lLS = mBuffers[0]->GetPacketQty();
    size_t lGS = lLS * mBuffers.size();

    assert(0 < lLS);

    Thread_OpenCL::Processing_Queue(&lGS, &lLS, mEvents + aIndex);
}

// CRITICAL_PATH
//
// Thread  Worker
//
// Processing_Queue ==> Processing_Wait
void Thread_Functions_OpenCL::Processing_Wait(unsigned int aIndex)
{
    assert(EVENT_QTY > aIndex);

    assert(NULL != mEvents[aIndex]);

    Thread_OpenCL::Processing_Wait(mEvents[aIndex], mKernel);

    mEvents[aIndex] = NULL;
}

void Thread_Functions_OpenCL::Prepare_Internal()
{
    assert(   0 < mAdapters.size());
    assert(   0 == mBuffers.size());
    assert(NULL != mKernel        );
    assert(NULL != mProcessor     );

    assert(NULL == mCommandQueue);
    assert(NULL == mProgram     );

    Processor_OpenCL * lProcessor = dynamic_cast<Processor_OpenCL *>(mProcessor);
    assert(NULL != lProcessor);

    mProgram = lProcessor->Program_Create(&mKernelFunctions);
    assert(NULL != mProgram);

    // Processor_Internal::CommandQueue_Create ==> OCLW_ReleaseCommandQueue  See Release
    mCommandQueue = lProcessor->CommandQueue_Create(mKernel->IsProfilingEnabled());
    assert(NULL != mCommandQueue);

    mKernel->SetCommandQueue(mCommandQueue);

    // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Release
    mKernel_CL = OCLW_CreateKernel(mProgram, "Filter");
    assert(NULL != mKernel_CL);

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        assert(NULL != mAdapters[i]);

        Adapter_Windows * lAdapter = dynamic_cast<Adapter_Windows *>(mAdapters[i]);
        assert(NULL != lAdapter);

        lAdapter->Buffers_Allocate(mCommandQueue, mKernel_CL, &mBuffers);
    }
}

void Thread_Functions_OpenCL::Release()
{
    Thread_Functions::Release();

    Thread_OpenCL::Release(mKernel);
}

void Thread_Functions_OpenCL::Run_Start()
{
    assert(0 < mBuffers.size());

    unsigned int i = 0;

    for (i = 0; i < mBuffers.size(); i++)
    {
        assert(NULL != mBuffers[i]);

        Buffer_Data_OpenCL * lBuffer = dynamic_cast<Buffer_Data_OpenCL *>(mBuffers[i]);
        assert(NULL != lBuffer);

        assert(NULL != lBuffer->mMem);

        OCLW_SetKernelArg(mKernel_CL, i, sizeof(cl_mem), &lBuffer->mMem);
    }

    for (i = 0; i < EVENT_QTY; i++)
    {
        Processing_Queue(i);
    }
}
