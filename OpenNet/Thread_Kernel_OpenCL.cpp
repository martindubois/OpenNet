
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Kernel_OpenCL.h

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

#include "Thread_Kernel_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aAdapter   [-K-;RW-]
// aKernel    [-K-;RW-]
// aProgram   [-K-;RW-]
// aDebugLog  [-K-;EW-]
Thread_Kernel_OpenCL::Thread_Kernel_OpenCL(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, cl_program aProgram, KmsLib::DebugLog * aDebugLog)
    : Thread_Kernel( aProcessor, aAdapter, aKernel, aDebugLog )
{
    assert(NULL != aProcessor);
    assert(NULL != aAdapter  );
    assert(NULL != aKernel   );
    assert(NULL != aProgram  );
    assert(NULL != aDebugLog );

    SetProgram(aProgram);
}

// aProgram [-K-;RW-] The corresponding cl_program
void Thread_Kernel_OpenCL::SetProgram(cl_program aProgram)
{
    assert(NULL != aProgram);

    assert(NULL == mProgram);

    mProgram = aProgram;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Kernel_OpenCL::Prepare_Internal()
{
    assert(   0 < mAdapters.size());
    assert(   0 == mBuffers.size());
    assert(NULL != mKernel        );
    assert(NULL != mProcessor     );

    assert(NULL == mCommandQueue);

    Thread_OpenCL::Prepare(dynamic_cast<Processor_OpenCL *>(mProcessor), &mAdapters, mKernel, &mBuffers);
}

void Thread_Kernel_OpenCL::Processing_Queue(unsigned int aIndex)
{
    assert(OPEN_NET_BUFFER_QTY > aIndex);

    assert(NULL != mCommandQueue);
    assert(NULL != mKernel_CL   );

    assert(NULL != mBuffers[aIndex]);

    Buffer_Data_OpenCL * lBuffer = dynamic_cast< Buffer_Data_OpenCL * >(mBuffers[aIndex]);
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

    Thread_OpenCL::Processing_Queue(&lGS, NULL, &lBuffer->mEvent);
}

void Thread_Kernel_OpenCL::Processing_Wait(unsigned int aIndex)
{
    assert(OPEN_NET_BUFFER_QTY > aIndex);

    assert(NULL != mBuffers[aIndex]);

    Buffer_Data_OpenCL * lBuffer = dynamic_cast<Buffer_Data_OpenCL *>(mBuffers[aIndex]);
    assert(NULL != lBuffer        );
    assert(NULL != lBuffer->mEvent);

    Thread_OpenCL::Processing_Wait(lBuffer->mEvent, mKernel);

    lBuffer->mEvent = NULL;
}

void Thread_Kernel_OpenCL::Release()
{
    Thread_OpenCL::Release(mKernel);
}

