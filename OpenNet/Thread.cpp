
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread.h

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
#ifdef _KMS_WINDOWS_
    #include "OCLW.h"
#endif

#include "Thread.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

#ifdef _KMS_WINDOWS_
    static uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam);
#endif

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-] The processor associated to the thread
// aDebugLog  [-K-;RW-] The DebugLog instance
Thread::Thread(Processor_Internal * aProcessor, KmsLib::DebugLog * aDebugLog)
    : mDebugLog    (aDebugLog )
    , mKernel      (NULL      )
    , mProcessor   (aProcessor)
    #ifdef _KMS_WINDOWS_
        , mCommandQueue(NULL)
        , mKernel_CL   (NULL)
        , mProgram     (NULL)
    #endif
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );

    SetPriority(PRIORITY_CRITICAL);
}

// aAdapter [-K-;RW-] The adapter to add
void Thread::AddAdapter(Adapter_Internal * aAdapter)
{
    assert(NULL != aAdapter);

    mAdapters.push_back(aAdapter);
}

OpenNet::Kernel * Thread::GetKernel()
{
    return mKernel;
}

OpenNet::Status Thread::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    assert(NULL != aOut         );
    assert(   0 <  aOutSize_byte);

    assert(NULL != mKernel);

    return mKernel->GetStatistics(aOut, aOutSize_byte, aInfo_byte, aReset);
}

OpenNet::Status Thread::ResetStatistics()
{
    assert(NULL != mKernel);

    return mKernel->ResetStatistics();
}

// aKernel  [-K-;RW-] The Kernel
void Thread::SetKernel(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel );

    assert(NULL == mKernel );

    mKernel  = aKernel ;
}

void Thread::Delete()
{
    try
    {
        switch (GetState())
        {
        case STATE_INIT:
            break;

        case STATE_RUNNING:
            ThreadBase::Stop();
            // no break;

        case STATE_STOP_REQUESTED:
        case STATE_STOPPING      :
            Stop_Wait(NULL, NULL);
            break;

        default: assert(false);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
    }

    Release();

    delete this;
}

// Exception  KmsLib::Exception *  See Processor_Internal::CommandQueue_Create
//                                 See OCLW_CreateKernel
//                                 See Adapter_Internal::Buffers_Allocate
void Thread::Prepare()
{
    assert(   0 <  mAdapters.size());
    assert(   0 == mBuffers .size());
    assert(NULL != mKernel         );
    assert(NULL != mProcessor      );

    #ifdef _KMS_WINDOWS_

        assert(NULL == mCommandQueue   );

        // Processor_Internal::CommandQueue_Create ==> OCLW_ReleaseCommandQueue  See Release
        mCommandQueue = mProcessor->CommandQueue_Create(mKernel->IsProfilingEnabled());
        assert(NULL != mCommandQueue);

        mKernel->SetCommandQueue(mCommandQueue);

        // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Release
        mKernel_CL = OCLW_CreateKernel(mProgram, "Filter");
        assert(NULL != mKernel_CL);

        unsigned int i;

        for (i = 0; i < mAdapters.size(); i++)
        {
            assert(NULL != mAdapters[i]);

            mAdapters[i]->Buffers_Allocate(mCommandQueue, mKernel_CL, &mBuffers);
        }

        for (unsigned int i = 0; i < mBuffers.size(); i++)
        {
            assert(NULL != mBuffers[i]);

            mBuffers[i]->ResetMarkerValue();
        }

    #endif
}

// aTryToSolveHang [--O;--X]
// aContext        [--O;---]
//
// Exception  KmsLib::Exception *  See Stop_Wait_Zone0
// Threads  Apps
void Thread::Stop_Wait(TryToSolveHang aTryToSolveHang, void * aContext)
{
    assert(   0 < mAdapters.size());
    assert(   0 < mBuffers.size ());
    assert(NULL != mDebugLog      );

    unsigned int i;

    bool lRetB = false;

    switch (GetState())
    {
    case STATE_STOP_REQUESTED:
        if (NULL != aTryToSolveHang)
        {
            ThreadBase::Sleep_s(1);

            for (i = 0; i < 2990; i++)
            {
                if (STATE_STOP_REQUESTED != GetState())
                {
                    break;
                }

                aTryToSolveHang(aContext, (1 == mAdapters.size()) ? mAdapters[0] : NULL);

                ThreadBase::Sleep_ms(100);
            }

            lRetB = Wait(true, 1000);
        }
        else
        {
            lRetB = Wait(true, 300000);
        }
        break;

    case STATE_STOPPING:
        lRetB = Wait(true, 1000);
        break;

    default: assert(false);
    }

    assert(lRetB);

    for (i = 0; i < mAdapters.size(); i++)
    {
        mAdapters[i]->Buffers_Release();
    }

    for (i = 0; i < mBuffers.size(); i++)
    {
        assert(NULL != mBuffers[i]);

        delete mBuffers[i];
    }
}

#ifdef _KMS_WINDOWS_

    // aProgram [-K-;RW-] The corresponding cl_program
    void Thread::SetProgram(cl_program aProgram)
    {
        assert(NULL != aProgram);

        assert(NULL == mProgram);

        mProgram = aProgram;
    }

#endif

// Internal
/////////////////////////////////////////////////////////////////////////////

// CRITICAL PATH
//
// Thread  Worker
unsigned int Thread::Run()
{
    assert(   0 <  mAdapters.size());
    assert(NULL != mDebugLog       );

    unsigned int lResult = __LINE__;

    try
    {
        Run_Start();

        unsigned int i;

        for (i = 0; i < mAdapters.size(); i++)
        {
            mAdapters[i]->Start();
        }

        Run_Loop();

        for (i = 0; i < mAdapters.size(); i++)
        {
            mAdapters[i]->Stop();
        }

        Run_Wait();

        lResult = 0;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
        lResult = __LINE__;
    }

    return lResult;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

Thread::~Thread()
{
}

void Thread::Release()
{
    // printf(__FUNCTION__ "()\n");

    assert(NULL != mDebugLog);

    #ifdef _KMS_WINDOWS_

        if (NULL != mCommandQueue)
        {
            mKernel->ResetCommandQueue();

            try
            {
                OCLW_ReleaseCommandQueue(mCommandQueue);
            }
            catch (...)
            {
                mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
            }

            if (NULL != mKernel_CL)
            {
                try
                {
                    OCLW_ReleaseKernel(mKernel_CL);
                }
                catch (...)
                {
                    mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
                }
            }
        }

    #endif
}

// aIndex  The index passed to Processing_Queue and Processing_Wait
//
// CRITICAL PATH - Buffer
void Thread::Run_Iteration(unsigned int aIndex)
{
    Processing_Wait (aIndex);
    Processing_Queue(aIndex);
}

#ifdef _KMS_WINDOWS_

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
    void Thread::Processing_Queue(const size_t * aGlobalSize, const size_t * aLocalSize, cl_event * aEvent)
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

    // aEvent [D--;RW-] The cl_event Processing_Queue created
    //
    // CRITICAL PATH - Buffer
    //
    // Processing_Queue ==> Processing_Wait
    void Thread::Processing_Wait(cl_event aEvent)
    {
        assert(NULL != aEvent);

        assert(NULL != mKernel);

        OCLW_WaitForEvents(1, &aEvent);

        if (mKernel->IsProfilingEnabled())
        {
            uint64_t lQueued = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_QUEUED);
            uint64_t lSubmit = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_SUBMIT);
            uint64_t lStart  = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_START );
            uint64_t lEnd    = GetEventProfilingInfo(aEvent, CL_PROFILING_COMMAND_END   );

            mKernel->AddStatistics(lQueued, lSubmit, lStart, lEnd);
        }

        OCLW_ReleaseEvent(aEvent);
    }

#endif

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_TIMEOUT
// Thread     Worker
void Thread::Run_Wait()
{
    assert(   0 <  mAdapters.size());
    assert(NULL != mDebugLog       );

    for (unsigned int i = 0; i < 3000; i++)
    {
        unsigned int lBufferCount = 0;

        for (unsigned int j = 0; j < mAdapters.size(); j++)
        {
            OpenNet::Adapter::State lState;

            OpenNet::Status lStatus = mAdapters[j]->GetState(&lState);
            if (OpenNet::STATUS_OK == lStatus)
            {
                lBufferCount += lState.mBufferCount;
            }
        }

        if (0 >= lBufferCount)
        {
            return;
        }

        ThreadBase::Sleep_ms(100);
    }

    // TODO  OpenNet.Adapter_Internal.Error_Handling
    //       Low - This is a big problem because the driver still use GPU
    //       buffer and the application is maybe going to release them.

    mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
        "The driver did not release the buffers in time", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

#ifdef _KMS_WINDOWS_

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

#endif
