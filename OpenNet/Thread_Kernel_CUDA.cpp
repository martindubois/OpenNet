
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_Kernel_CUDA.h

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "Adapter_Windows.h"
#include "Buffer_Data_CUDA.h"
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
    , Thread_CUDA( aModule )
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

    Thread_CUDA::Prepare( & mAdapters, & mBuffers );

    unsigned int lArgCount    = mKernel->GetArgumentCount();
    unsigned int lBufferCount = mBuffers.size();

    assert(                   1 <= lArgCount    );
    assert(                   0 <  lBufferCount );
    assert( OPEN_NET_BUFFER_QTY >= lBufferCount );

    mArguments = new void * [ lBufferCount * lArgCount ];
    assert( NULL != mArguments );

    memset( mArguments, 0, sizeof( void * ) * lBufferCount * lArgCount );

    for ( unsigned int i = 0; i < lBufferCount; i ++ )
    {
        Buffer_Data_CUDA * lBuffer = dynamic_cast< Buffer_Data_CUDA * >( mBuffers[ i ] );
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

    Thread_CUDA::Processing_Queue( mKernel, & lGS, NULL, lArguments );
}

void Thread_Kernel_CUDA::Processing_Wait(unsigned int aIndex)
{
    Thread_CUDA::Processing_Wait();
}

void Thread_Kernel_CUDA::Release()
{
    assert(NULL != mKernel);

    Thread_CUDA::Release( mKernel );
}
