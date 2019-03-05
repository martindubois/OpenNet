
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data.cpp

#define __CLASS__ "Buffer_Data::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ===== OpenNet ============================================================
#include "Buffer_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aPacketQty       The number of packet into the buffer
// aEvent [-K-;RW-] The event associated to the buffer
Buffer_Data::Buffer_Data(unsigned int aPacketQty, Event * aEvent)
    : mEvent      (aEvent    )
    , mPacketQty  (aPacketQty)
    , mMarkerValue(         0)
{
    assert( NULL != aEvent    );
    assert(    0 < aPacketQty );
}

// Exception  KmsLib::Exception *
Buffer_Data::~Buffer_Data()
{
}

void Buffer_Data::ResetMarkerValue()
{
    mMarkerValue = 0;
}
