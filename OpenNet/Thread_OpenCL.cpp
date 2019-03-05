
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet ============================================================
#include "Adapter_Windows.h"
#include "OCLW.h"

#include "Thread_OpenCL.h"

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam);

// Public
/////////////////////////////////////////////////////////////////////////////

Thread_OpenCL::~Thread_OpenCL()
{
}

// Protected
/////////////////////////////////////////////////////////////////////////////

Thread_OpenCL::Thread_OpenCL()
    : mCommandQueue(NULL)
    , mKernel_CL   (NULL)
    , mProgram     (NULL)
{
}

// aProcessor [---;RW-]
// aAdapters  [---;RW-]
// aKernel    [---;RW-]
// aBuffers   [---;RW-]
void Thread_OpenCL::Prepare(Processor_OpenCL * aProcessor, Adapter_Vector * aAdapters, OpenNet::Kernel * aKernel, Buffer_Data_Vector * aBuffers)
{
    assert(NULL != aKernel   );
    assert(NULL != aProcessor);

    // Processor_Internal::CommandQueue_Create ==> OCLW_ReleaseCommandQueue  See Release
    mCommandQueue = aProcessor->CommandQueue_Create(aKernel->IsProfilingEnabled());
    assert(NULL != mCommandQueue);

    aKernel->SetCommandQueue(mCommandQueue);

    // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Release
    mKernel_CL = OCLW_CreateKernel(mProgram, "Filter");
    assert(NULL != mKernel_CL);

    for (unsigned int i = 0; i < aAdapters->size(); i++)
    {
        assert(NULL != (*aAdapters)[i]);

        Adapter_Windows * lAdapter = dynamic_cast<Adapter_Windows *>((*aAdapters)[i]);
        assert(NULL != lAdapter);

        lAdapter->Buffers_Allocate(mCommandQueue, mKernel_CL, aBuffers);
    }
}

// aEvent      [---;RW-]
// aGlobalSize [---;R--]
// aLocalSize  [--O;R--]
//
// Processing_Queue ==> Processing_Wait

// CRITICAL PATH  Processing
//                1 / iteration
void Thread_OpenCL::Processing_Queue( Event_OpenCL * aEvent, const size_t * aGlobalSize, const size_t * aLocalSize )
{
    assert(NULL != aEvent     );
    assert(NULL != aGlobalSize);

    assert(NULL != mCommandQueue);
    assert(NULL != mKernel_CL   );

    size_t lGO = 0;

    // OCLW_EnqueueNDRangeKernel ==> OCLW_ReleaseEvent  See Processing_Wait
    OCLW_EnqueueNDRangeKernel(mCommandQueue, mKernel_CL, 1, &lGO, aGlobalSize, aLocalSize, 0, NULL, aEvent->mEvent);

    OCLW_Flush(mCommandQueue);
}

// aKernel [---;RW-]
void Thread_OpenCL::Release(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    if (NULL != mCommandQueue)
    {
        aKernel->ResetCommandQueue();

        try
        {
            OCLW_ReleaseCommandQueue(mCommandQueue);
        }
        catch (...)
        {
        }

        if (NULL != mKernel_CL)
        {
            try
            {
                OCLW_ReleaseKernel(mKernel_CL);
            }
            catch (...)
            {
            }
        }
    }
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aEvent [---;R--]
// aParam
//
// Return  This method return the retrieved information
//
// Exception  KmsLib::Exception *  See OCLW_GetEventProfilingInfo
// Thread     Worker

// CRITICAL PATH  Processing.Profiling
//                4 / iteration
uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam)
{
    assert(NULL != aEvent);

    uint64_t lResult;

    OCLW_GetEventProfilingInfo(aEvent, aParam, sizeof(lResult), &lResult);

    return lResult;
}
