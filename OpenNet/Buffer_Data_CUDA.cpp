
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_CUDA.cpp

#define __CLASS__ "Buffer_Data_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ===== OpenNet ============================================================
#include "CUW.h"

#include "Buffer_Data_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProfiling         Set to true when profiling data must be captured
// aContext [-K-;RW-] The CUDA context
// aMem_DA  [DK-;RW-] The pointer to the device memory
// aPacketQty         The number of packet the buffer contains
Buffer_Data_CUDA::Buffer_Data_CUDA( bool aProfiling, CUcontext aContext, CUdeviceptr aMem_DA, unsigned int aPacketQty )
    : Buffer_Data( aPacketQty, static_cast< Event * >( & mEvent_CUDA ) )
    , mContext   ( aContext   )
    , mEvent_CUDA( aProfiling )
    , mMemory_DA ( aMem_DA    )
{
    assert( NULL != aContext   );
    assert(    0 != aMem_DA    );
    assert(    0 <  aPacketQty );
}

// ===== Buffer_Data ========================================================

Buffer_Data_CUDA::~Buffer_Data_CUDA()
{
    assert( NULL != mContext   );
    assert(    0 != mMemory_DA );

    CUW_CtxSetCurrent( mContext );
    
    // CUW_MemAlloc ==> CUW_MemFree
    CUW_MemFree( mMemory_DA );
}
