
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread.cpp

#define __CLASS__ "Thread::"

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
#include "Thread.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-] The processor associated to the thread
// aDebugLog  [-K-;RW-] The DebugLog instance
Thread::Thread(Processor_Internal * aProcessor, KmsLib::DebugLog * aDebugLog)
    : mDebugLog    (aDebugLog )
    , mKernel      (NULL      )
    , mProcessor   (aProcessor)
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
        mDebugLog->Log(__FILE__, __CLASS__ "Delete", __LINE__);
        mDebugLog->Log(eE);
    }

    Release();

    // printf( __CLASS__ "Delete - delete 0x%lx (this)\n", reinterpret_cast< uint64_t >( this ) );

    delete this;
}

// Exception  KmsLib::Exception *
void Thread::Prepare()
{
    // printf( __CLASS__ "Prepare()\n" );

    assert(   0 <  mAdapters.size());
    assert(   0 == mBuffers .size());

    for (unsigned int i = 0; i < mBuffers.size(); i++)
    {
        assert(NULL != mBuffers[i]);

        mBuffers[i]->ResetMarkerValue();
    }
}

// aTryToSolveHang [--O;--X]
// aContext        [--O;---]
//
// Exception  KmsLib::Exception *  See Stop_Wait_Zone0
// Threads  Apps
void Thread::Stop_Wait(TryToSolveHang aTryToSolveHang, void * aContext)
{
    // printf( __CLASS__ "Stop_Wait( ,  )\n" );

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

        printf( __CLASS__ "Stop_Wait - delete 0x%lx (mBuffers[ %u ])\n", reinterpret_cast< uint64_t >( mBuffers[ i ] ), i );

        delete mBuffers[i];
    }
}

// Internal
/////////////////////////////////////////////////////////////////////////////

// CRITICAL PATH
//
// Thread  Worker
unsigned int Thread::Run()
{
    // printf( __CLASS__ "Run()\n" );

    assert(   0 <  mAdapters.size());
    assert(NULL != mDebugLog       );

    unsigned int lResult = __LINE__;

    try
    {
        Run_Start();

        unsigned int i;

        // printf( __CLASS__ "Run - %u adapters\n", static_cast< unsigned int >( mAdapters.size() ) );

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
        eE->Write( stdout );
        mDebugLog->Log(__FILE__, __CLASS__ "Run", __LINE__);
        mDebugLog->Log(eE);
        lResult = __LINE__;
    }

    // printf( __CLASS__ "Run - Return %u\n", lResult );
    return lResult;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

Thread::~Thread()
{
}

// aIndex  The index passed to Processing_Queue and Processing_Wait
//
// CRITICAL PATH - Buffer
void Thread::Run_Iteration(unsigned int aIndex)
{
    // printf( __CLASS__ "Run_Iteration( %u )\n", aIndex );

    Processing_Wait (aIndex);
    Processing_Queue(aIndex);
}

// Exception  KmsLib::Exception *  CODE_TIMEOUT
// Thread     Worker
void Thread::Run_Wait()
{
    // printf( __CLASS__ "Run_Wait()\n" );

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

    mDebugLog->Log(__FILE__, __CLASS__ "Run_Wait", __LINE__);
    throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
        "The driver did not release the buffers in time", NULL, __FILE__, __CLASS__ "Run_Wait", __LINE__, 0);
}
