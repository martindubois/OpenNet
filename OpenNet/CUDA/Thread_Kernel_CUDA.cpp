
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Thread_Kernel_CUDA.h

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== OpenNet/CUDA =======================================================
#include "Buffer_CUDA.h"
#include "CUDAW.h"
#include "Processor_CUDA.h"

#include "Thread_Kernel_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aAdapter   [-K-;RW-]
// aKernel    [-K-;RW-]
// aModule    [-K-;RW-]
// aDebugLog  [-K-;EW-]
Thread_Kernel_CUDA::Thread_Kernel_CUDA(Processor_Internal * aProcessor, Adapter_Internal * aAdapter, OpenNet::Kernel * aKernel, CUmodule aModule, KmsLib::DebugLog * aDebugLog)
    : Thread_Kernel( aProcessor, aAdapter, aKernel, aDebugLog )
    , Thread_CUDA  ( aProcessor, aModule )
{
    assert(NULL != aProcessor);
    assert(NULL != aAdapter  );
    assert(NULL != aKernel   );
    assert(NULL != aModule   );
    assert(NULL != aDebugLog );
}

// ===== Thread =============================================================

void Thread_Kernel_CUDA::Prepare()
{
    assert( NULL == mArguments );
    assert( NULL != mKernel    );

    Thread_CUDA::Prepare( & mAdapters, & mBuffers, mKernel->IsProfilingEnabled(), mKernel );

    mQueueDepth = mBuffers.size();

    unsigned int lArgCount    = mKernel->GetArgumentCount();

    assert(                   1 <= lArgCount   );
    assert(                   0 <  mQueueDepth );
    assert( OPEN_NET_BUFFER_QTY >= mQueueDepth );

    mArguments = new void * [ mQueueDepth * lArgCount ];
    assert( NULL != mArguments );

    memset( mArguments, 0, sizeof( void * ) * mQueueDepth * lArgCount );

    for ( unsigned int i = 0; i < mQueueDepth; i ++ )
    {
        Buffer_CUDA * lBuffer = dynamic_cast< Buffer_CUDA * >( mBuffers[ i ] );
        assert( NULL != lBuffer             );
        assert(    0 != lBuffer->mMemory_DA );

        mArguments[ i * lArgCount ] = & lBuffer->mMemory_DA;
    }
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Thread =============================================================

void Thread_Kernel_CUDA::Processing_Queue( unsigned int aIndex )
{
    assert( OPEN_NET_BUFFER_QTY > aIndex );

    assert( NULL != mBuffers[ aIndex ] );
    assert( NULL != mKernel            );

    unsigned int lArgCount = mKernel->GetArgumentCount();
    assert( 1 <= lArgCount );

    void * * lArguments = mArguments + ( aIndex * lArgCount );

    mKernel->SetUserKernelArgs( lArguments );

    size_t lGS = mBuffers[ aIndex ]->GetPacketQty();

    assert(0 < lGS);

    Thread_CUDA::Processing_Queue( mKernel, mBuffers[ aIndex ]->GetEvent(), & lGS, NULL, lArguments );
}

void Thread_Kernel_CUDA::Run_Start()
{
    Thread_CUDA  ::Run_Start();
    Thread_Kernel::Run_Start();
}

void Thread_Kernel_CUDA::Release()
{
    assert(NULL != mKernel    );

    Thread_CUDA::Release( mKernel );
}
