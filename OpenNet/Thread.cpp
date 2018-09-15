
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Thread.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam);

// ===== Entry point ========================================================
static DWORD WINAPI Run(LPVOID aParameter);

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-] The processor associated to the thread
// aDebugLog  [-K-;RW-] The DebugLog instance
Thread::Thread(Processor_Internal * aProcessor, KmsLib::DebugLog * aDebugLog)
    : mCommandQueue(NULL      )
    , mDebugLog    (aDebugLog )
    , mKernel      (NULL      )
    , mKernel_CL   (NULL      )
    , mProcessor   (aProcessor)
    , mProgram     (NULL      )
    , mState       (STATE_INIT)
    , mThread      (NULL      )
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );

    // InitializeCriticalSection ==> DeleteCriticalSection  See Thread::~Thread
    InitializeCriticalSection(&mZone0);
}

Thread::~Thread()
{
    try
    {
        EnterCriticalSection(&mZone0);

            switch (mState)
            {
            case STATE_INIT:
                break;

            case STATE_RUNNING :
            case STATE_STARTING:
            case STATE_STOPPING:
                Stop_Request_Zone0();
                Stop_Wait_Zone0   (NULL, NULL);
                // Stop_Wait_Zone0 release the gate when it throw an
                // exception.
                break;

            default: assert(false);
            }

        LeaveCriticalSection(&mZone0);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }

    // InitializeCriticalSection ==> DeleteCriticalSection  See Thread::Thread
    DeleteCriticalSection(&mZone0);

    Release();
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

// aProgram [-K-;RW-] The corresponding cl_program
void Thread::SetProgram(cl_program aProgram)
{
    assert(NULL != aProgram);

    assert(NULL == mProgram);

    mProgram = aProgram;
}

// Exception  KmsLib::Exception *  See Processor_Internal::CommandQueue_Create
//                                 See OCLW_CreateKernel
//                                 See Adapter_Internal::Buffers_Allocate
void Thread::Prepare()
{
    assert(   0 <  mAdapters.size());
    assert(   0 == mBuffers .size());
    assert(NULL == mCommandQueue   );
    assert(NULL != mKernel         );
    assert(NULL != mProcessor      );

    // Processor_Internal::CommandQueue_Create ==> OCLW_ReleaseCommandQueue  See Release
    mCommandQueue = mProcessor->CommandQueue_Create(mKernel->IsProfilingEnabled());
    assert(NULL != mCommandQueue);

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
}

// Exception  KmsLib::Exception *  CODE_THREAD_ERROR
//                                 See State_Change
// Threads  Apps
void Thread::Start()
{
    assert(NULL != mDebugLog);
    assert(NULL == mThread  );
    assert(0    == mThreadId);

    State_Change(STATE_INIT, STATE_START_REQUESTED);

    mThread = CreateThread(NULL, 0, ::Run, this, 0, &mThreadId);
    if (NULL == mThread)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_THREAD_ERROR,
            "CreateThread( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }
}

// Threads  Apps
void Thread::Stop_Request()
{
    EnterCriticalSection(&mZone0);
        Stop_Request_Zone0();
    LeaveCriticalSection(&mZone0);
}

// aTryToSolveHang [--O;--X]
// aContext        [--O;---]
//
// Exception  KmsLib::Exception *  See Stop_Wait_Zone0
// Threads  Apps
void Thread::Stop_Wait(TryToSolveHang aTryToSolveHang, void * aContext)
{
    assert(0 < mAdapters.size());
    assert(0 < mBuffers.size ());

    EnterCriticalSection(&mZone0);
        Stop_Wait_Zone0(aTryToSolveHang, aContext);
    LeaveCriticalSection(&mZone0);

    unsigned int i;

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

// Internal
/////////////////////////////////////////////////////////////////////////////

// CRITICAL PATH
//
// Thread  Worker
void Thread::Run()
{
    assert(   0 <  mAdapters.size());
    assert(NULL != mDebugLog       );

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }

    try
    {
        State_Change(STATE_START_REQUESTED, STATE_STARTING);

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

    mState = STATE_STOPPING;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

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

// aIndex  The index passed to Processing_Queue and Processing_Wait
//
// CRITICAL PATH - Buffer
void Thread::Run_Iteration(unsigned int aIndex)
{
    Processing_Wait (aIndex);
    Processing_Queue(aIndex);
}

// aFrom  The state to leave
// aTo    The state to enter
//
// Exception  KmsLib::Exception *  CODE_STATE_ERROR
// Threads    Apps, Worker
void Thread::State_Change(State aFrom, State aTo)
{
    assert(STATE_QTY > aFrom);
    assert(STATE_QTY > aTo  );

    assert(NULL != mDebugLog);

    bool lOK = false;

    EnterCriticalSection(&mZone0);

        assert(STATE_QTY > mState);

        if (mState == aFrom)
        {
            lOK    = true;
            mState = aTo ;
        }

    LeaveCriticalSection(&mZone0);

    if (!lOK)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_STATE_ERROR,
            "Invalid state transition", NULL, __FILE__, __FUNCTION__, __LINE__, aTo);
    }
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Thread::Release()
{
    assert(NULL != mDebugLog);

    if (NULL != mCommandQueue)
    {
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
}

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

        Sleep(100);
    }

    // TODO  OpenNet.Adapter_Internal.Error_Handling
    //       This is a big program because the driver still use GPU buffer
    //       and the application is maybe going to release them.

    mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
        "The driver did not release the buffers in time", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
}

// Threads  Apps
void Thread::Stop_Request_Zone0()
{
    switch (mState)
    {
    case STATE_RUNNING :
    case STATE_STARTING:
        mState = STATE_STOP_REQUESTED;
        break;

    case STATE_STOPPING :
        break;

    default: assert(false);
    }
}

// aTryToSolveHang [--O;--X]
// aContext        [--O;---]
//
// This method release the Zone0 when it throw an exception.
//
// Exception  KmsLib::Exception *  CODE_THREAD_ERROR
// Threads  Apps
void Thread::Stop_Wait_Zone0(TryToSolveHang aTryToSolveHang, void * aContext)
{
    assert(   0 <  mAdapters.size());
    assert(NULL != mDebugLog       );
    assert(NULL != mThread         );

    switch (mState)
    {
    case STATE_STOP_REQUESTED:
    case STATE_STOPPING      :
        DWORD lRet;

        if (NULL != aTryToSolveHang)
        {
            lRet = Wait_Zone0(1000);

            if (WAIT_OBJECT_0 == lRet) { break; }

            for (unsigned int i = 0; i < 2990; i ++)
            {
                aTryToSolveHang(aContext, ( 1 == mAdapters.size() ) ? mAdapters[ 0 ] : NULL);

                lRet = Wait_Zone0(100);

                if (WAIT_OBJECT_0 == lRet) { break; }
            }
        }
        else
        {
            lRet = Wait_Zone0(300000);
        }

        if (WAIT_OBJECT_0 == lRet) { break; }

        // TODO  OpenNet.Adapter_Internal.ErrorHandling
        //       This case is a big problem. Terminating the thread
        //       interracting with the GPU may let the system in an instabla
        //       state. Worst, in this case the drive still use the GPU
        //       buffer.

        LeaveCriticalSection(&mZone0);
            if (!TerminateThread(mThread, __LINE__))
            {
                mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
                throw new KmsLib::Exception(KmsLib::Exception::CODE_THREAD_ERROR,
                    "TerminateThread( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
            }
        EnterCriticalSection(&mZone0);
        break;

    default: assert(false);
    }

    mState = STATE_INIT;

    BOOL lRetB = CloseHandle(mThread);
    assert(lRetB);
    (void)(lRetB);

    mThread   = NULL;
    mThreadId = 0   ;
}

// aTimeout_ms
//
// Return  See WaitForSingleObject
unsigned int Thread::Wait_Zone0(unsigned int aTimeout_ms)
{
    assert(0 < aTimeout_ms);

    assert(NULL != mThread);

    unsigned int lResult;

    LeaveCriticalSection(&mZone0);
        lResult = WaitForSingleObject(mThread, aTimeout_ms);
    EnterCriticalSection(&mZone0);

    return lResult;
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

// ===== Entry point ========================================================

// aParameter [---;RW-] The this pointer
//
// Return  This method always return 0
//
// Thread  Worker (Entry point)
DWORD WINAPI Run(LPVOID aParameter)
{
    assert(NULL != aParameter);

    Thread * lThis = reinterpret_cast<Thread *>(aParameter);

    lThis->Run();

    return 0;
}
