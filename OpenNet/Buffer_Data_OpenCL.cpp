
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Buffer_Data_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aMem [DK-;RW-] The cl_mem instance describing the buffer
// aPacketQty     The number of packet the buffer contains
Buffer_Data_OpenCL::Buffer_Data_OpenCL(cl_mem aMem, unsigned int aPacketQty)
    : Buffer_Data( aPacketQty )
    , mEvent(NULL)
    , mMem  (aMem)
{
    assert(NULL != aMem      );
    assert(   0 <  aPacketQty);
}

// ===== Buffer_Data ========================================================

Buffer_Data_OpenCL::~Buffer_Data_OpenCL()
{
    assert(NULL != mMem);

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See ?
    OCLW_ReleaseMemObject(mMem);
}
