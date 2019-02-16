
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_CUDA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "CUW.h"

#include "Buffer_Data_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aMem_DA     The pointer to the device memory
// aPacketQty  The number of packet the buffer contains
Buffer_Data_CUDA::Buffer_Data_CUDA( CUdeviceptr aMem_DA, unsigned int aPacketQty )
    : Buffer_Data( aPacketQty )
    , mMemory_DA( aMem_DA )
{
    assert( 0 != aMem_DA    );
    assert( 0 <  aPacketQty );
}

// ===== Buffer_Data ========================================================

Buffer_Data_CUDA::~Buffer_Data_CUDA()
{
    assert( 0 != mMemory_DA );

    // CUW_MemAlloc ==> CUW_MemFree
    CUW_MemFree( mMemory_DA );
}
