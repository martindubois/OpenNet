
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Thread_Functions_CUDA.h

#define __CLASS__ "Thread_Functions_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== Commmon ============================================================
#include "../Common/Constants.h"

// ===== OpenNet/CUDA =======================================================
#include "Buffer_CUDA.h"
#include "CUW.h"

#include "Thread_Functions_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
Thread_Functions_CUDA::Thread_Functions_CUDA(Processor_Internal * aProcessor, bool aProfilingEnabled, KmsLib::DebugLog * aDebugLog)
    : Thread_Functions( aProcessor, aProfilingEnabled, aDebugLog )
    , Thread_CUDA     ( aProcessor )
{
    for ( unsigned int i = 0; i < QUEUE_DEPTH; i ++ )
    {
        mEvents[ i ] = mEvent_CUDA + i;
    }
}

// ===== Thread =============================================================

void Thread_Functions_CUDA::Prepare()
{
    assert(NULL != mKernel    );
    assert(NULL == mModule    );
    assert(NULL != mProcessor );

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
    assert( NULL != lProcessor );

    // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See Release
    mModule = lProcessor->Module_Create( & mKernelFunctions, ADAPTER_NO_UNKNOWN );
    assert( NULL != mModule );

    Thread_CUDA::Prepare( & mAdapters, & mBuffers, false );

    Thread_Functions::Prepare();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Functions_CUDA::Processing_Queue(unsigned int aIndex)
{
    // printf( __CLASS__ "Processing_Queue( %u )\n", aIndex );

    assert(QUEUE_DEPTH > aIndex);

    assert( NULL != mArguments  );
    assert( NULL != mBuffers[0] );
    assert( NULL != mEvents[ aIndex ] );
    assert( NULL != mKernel     );

    size_t lLS = mBuffers[0]->GetPacketQty();
    size_t lGS = lLS * mBuffers.size();

    assert( 0 < lGS );

    Thread_CUDA::Processing_Queue( mKernel, mEvents[ aIndex ], & lGS, & lLS, mArguments );
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

        Buffer_CUDA * lBuffer = dynamic_cast< Buffer_CUDA * >( mBuffers[ i ] );
        assert( NULL != lBuffer );

        mArguments[ i ] = & lBuffer->mMemory_DA;
    }

    Thread_Functions::Run_Start();
}

void Thread_Functions_CUDA::Release()
{
    assert( NULL != mKernel );
    assert( NULL != mModule );

    Thread_CUDA::Release( mKernel );

    // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See Prepare
    CUW_ModuleUnload( mModule );
}
