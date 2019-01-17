
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
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );
}

// ===== Thread =============================================================

void Thread_Functions_OpenCL::Prepare()
{
    assert(   0 < mAdapters.size());
    assert(NULL != mKernel        );
    assert(NULL != mProcessor     );

    assert(NULL == mProgram     );

    Processor_OpenCL * lProcessor = dynamic_cast<Processor_OpenCL *>(mProcessor);
    assert(NULL != lProcessor);

    mProgram = lProcessor->Program_Create(&mKernelFunctions);
    assert(NULL != mProgram);

    Thread_OpenCL::Prepare(dynamic_cast<Processor_OpenCL *>(mProcessor), &mAdapters, mKernel, &mBuffers);
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

void Thread_Functions_OpenCL::Release()
{
    assert(NULL != mKernel);

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

    Thread_Functions::Run_Start();
}
