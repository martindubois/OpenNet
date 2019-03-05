
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions.h

#define __CLASS__ "Thread_Function::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== OpenNet ============================================================
#include "Event.h"

#include "Thread_Functions.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aProfilingEnabled
// aDebugLog  [-K-;RW-]
Thread_Functions::Thread_Functions(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
    : Thread(aProcessor, aDebugLog)
{
    assert(NULL != aProcessor);
    assert(NULL != aDebugLog );

    mQueueDepth = QUEUE_DEPTH;

    if (aProfilingEnabled)
    {
        mKernelFunctions.EnableProfiling();
    }
    else
    {
        mKernelFunctions.DisableProfiling();
    }

    SetKernel(&mKernelFunctions);
}

// aAdapter  [-K-;RW-]
// aFunction [---;R--]
void Thread_Functions::AddAdapter(Adapter_Internal * aAdapter, const OpenNet::Function & aFunction)
{
    assert(NULL !=   aAdapter  );
    assert(NULL != (&aFunction));

    mKernelFunctions.AddFunction(aFunction);

    Thread::AddAdapter(aAdapter);
}

void Thread_Functions::AddDispatchCode()
{
    unsigned int lAdapterCount = static_cast<unsigned int>(mAdapters.size());
    assert(0 < lAdapterCount);

    unsigned int * lBufferQty = new unsigned int[lAdapterCount];
    assert(NULL != lBufferQty);

    for (unsigned int i = 0; i < lAdapterCount; i++)
    {
        assert(NULL != mAdapters[i]);

        lBufferQty[i] = mAdapters[i]->GetBufferQty();
    }

    mKernelFunctions.AddDispatchCode(lBufferQty);

    // printf( __CLASS__ "AddDispatchCode - delete [] 0x%lx (lBufferQty)\n", reinterpret_cast< uint64_t >( lBufferQty ) );

    delete[] lBufferQty;
}

// ===== Thread =============================================================

void Thread_Functions::Prepare()
{
    assert( NULL != mKernel );

    for ( unsigned int i = 0; i < QUEUE_DEPTH; i ++ )
    {
        assert( NULL != mEvents[ i ] );

        mEvents[ i ]->Init( mKernel->IsProfilingEnabled() );
    }

    Thread::Prepare();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

void Thread_Functions::Processing_Wait( unsigned int aIndex )
{
    assert( QUEUE_DEPTH >  aIndex  );

    assert( NULL != mEvents[ aIndex ] );
    assert( NULL != mKernel           );

    Thread::Processing_Wait( mKernel, mEvents[ aIndex ] );
}

// ===== Thread =============================================================

void Thread_Functions::Release()
{
    assert(NULL != mProcessor);

    mProcessor->Thread_Release();
}
