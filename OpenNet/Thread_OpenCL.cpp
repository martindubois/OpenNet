
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

// aGlobalSize [---;R--]
// aLocalSize  [--O;R--]
// aEvent      [---;-W-] This methode return a newly created cl_event here.
//                       This event will be signaled at the end of
//                       processing. This event must be passed to
//                       Processing_Wait.
//
// CRITICAL PATH - Buffer
//
// Processing_Queue ==> Processing_Wait
void Thread_OpenCL::Processing_Queue(const size_t * aGlobalSize, const size_t * aLocalSize, cl_event * aEvent)
{
    assert(NULL != aGlobalSize);
    assert(NULL != aEvent     );

    assert(NULL != mCommandQueue);
    assert(NULL != mKernel_CL   );

    size_t lGO = 0;

    // OCLW_EnqueueNDRangeKernel ==> OCLW_ReleaseEvent  See Processing_Wait
    OCLW_EnqueueNDRangeKernel(mCommandQueue, mKernel_CL, 1, &lGO, aGlobalSize, aLocalSize, 0, NULL, aEvent);

    OCLW_Flush(mCommandQueue);
}

// aEvent  [D--;RW-] The cl_event Processing_Queue created
// aKernel [---;RW-]
//
// CRITICAL PATH - Buffer
//
// Processing_Queue ==> Processing_Wait
void Thread_OpenCL::Processing_Wait(cl_event aEvent, OpenNet::Kernel * aKernel)
{
    assert(NULL != aEvent );
    assert(NULL != aKernel);

    OCLW_WaitForEvents(1, &aEvent);

    if (aKernel->IsProfilingEnabled())
    {
        uint64_t lQueued = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_QUEUED);
        uint64_t lSubmit = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_SUBMIT);
        uint64_t lStart  = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_START );
        uint64_t lEnd    = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_END   );

        aKernel->AddStatistics(lQueued, lSubmit, lStart, lEnd);
    }

    OCLW_ReleaseEvent(aEvent);
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
// CRITICAL PATH - Buffer
//
// Exception  KmsLib::Exception *  See OCLW_GetEventProfilingInfo
// Thread     Worker
uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam)
{
    assert(NULL != aEvent);

    uint64_t lResult;

    OCLW_GetEventProfilingInfo(aEvent, aParam, sizeof(lResult), &lResult);

    return lResult;
}
