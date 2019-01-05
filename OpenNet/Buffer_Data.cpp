
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#ifdef _KMS_WINDOWS_
    #include "OCLW.h"
#endif

#include "Buffer_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Buffer_Data::~Buffer_Data()
{
    #ifdef _KMS_WINDOWS_
        assert(NULL != mMem);

        // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See ?
        OCLW_ReleaseMemObject(mMem);
    #endif
}

// Return  The next value to wait for
uint32_t Buffer_Data::GetMarkerValue()
{
    mMarkerValue++;

    return mMarkerValue;
}

// Return  The number of packet the buffer contains
unsigned int Buffer_Data::GetPacketQty() const
{
    assert(0 < mPacketQty);

    return mPacketQty;
}

void Buffer_Data::ResetMarkerValue()
{
    mMarkerValue = 0;
}

#ifdef _KMS_WINDOWS_

    // aMem [DK-;RW-] The cl_mem instance describing the buffer
    // aPacketQty     The number of packet the buffer contains
    Buffer_Data::Buffer_Data(cl_mem aMem, unsigned int aPacketQty) : mEvent(NULL), mMem(aMem), mPacketQty(aPacketQty), mMarkerValue(0)
    {
        assert(NULL != aMem      );
        assert(   0 <  aPacketQty);
    }

#endif
