
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Buffer_Data.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Buffer_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aMem [DK-;RW-] The cl_mem instance describing the buffer
// aPacketQty     The number of packet the buffer contains
Buffer_Data::Buffer_Data(cl_mem aMem, unsigned int aPacketQty) : mEvent(NULL), mMem(aMem), mPacketQty(aPacketQty), mMarkerValue(0)
{
    assert(NULL != aMem      );
    assert(   0 <  aPacketQty);
}

Buffer_Data::~Buffer_Data()
{
    assert(NULL != mMem);

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See ?
    OCLW_ReleaseMemObject(mMem);
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
