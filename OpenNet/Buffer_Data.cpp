
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
#include "Buffer_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aPacketQty  The number of packet into the buffer
Buffer_Data::Buffer_Data(unsigned int aPacketQty)
    : mPacketQty  (aPacketQty)
    , mMarkerValue(         0)
{
    assert(0 < aPacketQty);
}

Buffer_Data::~Buffer_Data()
{
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
