
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Buffer_CUDA.cpp

#define __CLASS__ "Buffer_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdint.h>
#include <stdio.h>

// ===== OpenNet/CUDA =======================================================
#include "CUW.h"

#include "Buffer_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProfiling         Set to true when profiling data must be captured
// aContext [-K-;RW-] The CUDA context
// aMem_DA  [DK-;RW-] The pointer to the device memory
// aPacketQty         The number of packet the buffer contains
Buffer_CUDA::Buffer_CUDA( bool aProfiling, CUcontext aContext, CUdeviceptr aMem_DA, unsigned int aPacketQty )
    : Buffer_Internal( aPacketQty, static_cast< Event * >( & mEvent_CUDA ) )
    , mContext   ( aContext   )
    , mEvent_CUDA( aProfiling )
    , mMemory_DA ( aMem_DA    )
{
    assert( NULL != aContext   );
    assert(    0 != aMem_DA    );
    assert(    0 <  aPacketQty );
}

// ===== Buffer_Internal ====================================================

Buffer_CUDA::~Buffer_CUDA()
{
    assert( NULL != mContext   );
    assert(    0 != mMemory_DA );

    CUW_CtxSetCurrent( mContext );
    
    // CUW_MemAlloc ==> CUW_MemFree
    CUW_MemFree( mMemory_DA );
}

void Buffer_CUDA::FetchBufferInfo()
{
    assert( NULL != mContext );

    CUW_CtxSetCurrent( mContext );

    Buffer_Internal::FetchBufferInfo();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer_Internal ====================================================

void Buffer_CUDA::Read( unsigned int aOffset_byte, void * aOut, unsigned int aOutSize_byte )
{
    assert( NULL != aOut          );
    assert(    0 <  aOutSize_byte );

    assert( 0 != mMemory_DA );

    CUW_MemcpyDtoH( aOut, mMemory_DA + aOffset_byte, aOutSize_byte );
}

void Buffer_CUDA::Write( unsigned int aOffset_byte, const void * aIn , unsigned int aInSize_byte )
{
    assert( NULL != aIn          );
    assert( 0    <  aInSize_byte );

    assert( 0 != mMemory_DA );

    CUW_MemcpyHtoD( mMemory_DA + aOffset_byte, aIn, aInSize_byte );
}

void Buffer_CUDA::Wait_Internal()
{
}
