
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Functions_CUDA.h

#define __CLASS__ "Thread_Functions_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "Buffer_Data_CUDA.h"

#include "Thread_Functions_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
Thread_Functions_CUDA::Thread_Functions_CUDA(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
    : Thread_Functions( aProcessor, aProfilingEnabled, aDebugLog )
    , Thread_CUDA     ( aProcessor )
{
}

// ===== Thread =============================================================

void Thread_Functions_CUDA::Prepare()
{
    assert(NULL != mKernel    );
    assert(NULL != mProcessor );

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
    assert( NULL != lProcessor );

    mModule = lProcessor->Module_Create( & mKernelFunctions );
    assert( NULL != mModule );

    Thread_CUDA::Prepare( & mAdapters, & mBuffers, EVENT_QTY );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Functions_CUDA::Processing_Queue(unsigned int aIndex)
{
    // printf( __CLASS__ "Processing_Queue( %u )\n", aIndex );

    assert(EVENT_QTY > aIndex);

    assert( NULL != mArguments  );
    assert( NULL != mBuffers[0] );
    assert( NULL != mKernel     );

    size_t lLS = mBuffers[0]->GetPacketQty();
    size_t lGS = lLS * mBuffers.size();

    assert( 0 < lGS );

    Thread_CUDA::Processing_Queue( mKernel, & lGS, & lLS, mArguments );
}

void Thread_Functions_CUDA::Processing_Wait(unsigned int aIndex)
{
    Thread_CUDA::Processing_Wait();
}

void Thread_Functions_CUDA::Run_Start()
{
    assert( NULL == mArguments );

    assert(0 < mBuffers.size());

    // new ==> delete  See Run_Wait
    mArguments = new void * [ mBuffers.size() ];
    assert( NULL != mArguments );

    memset( mArguments, 0, sizeof( void * ) * mBuffers.size() );

    for ( unsigned int i = 0; i < mBuffers.size(); i++)
    {
        assert( NULL != mBuffers[ i] );

        Buffer_Data_CUDA * lBuffer = dynamic_cast< Buffer_Data_CUDA * >( mBuffers[ i ] );
        assert( NULL != lBuffer );

        mArguments[ i ] = & lBuffer->mMemory_DA;
    }

    Thread_Functions::Run_Start();
}

void Thread_Functions_CUDA::Release()
{
    assert( NULL != mKernel );

    Thread_CUDA::Release( mKernel );
}
