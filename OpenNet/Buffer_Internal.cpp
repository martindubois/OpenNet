
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== OpenNet ============================================================
#include "Utils.h"

#include "Buffer_Internal.h"

// Static constants
/////////////////////////////////////////////////////////////////////////////

static const uint32_t EVENT_CLEAR = OPEN_NET_BUFFER_PROCESSED;

// Public
/////////////////////////////////////////////////////////////////////////////

// aPacketQty       The number of packet into the buffer
// aEvent [-K-;RW-] The event associated to the buffer
Buffer_Internal::Buffer_Internal(unsigned int aPacketQty, Event * aEvent)
    : mEvent      (aEvent    )
    , mPacketQty  (aPacketQty)
    , mMarkerValue(0         )
{
    assert(NULL != aEvent    );
    assert(   0 <  aPacketQty);

    mCache_Header = reinterpret_cast<const OpenNet_BufferHeader *>(mCache);
}

// Exception  KmsLib::Exception *
Buffer_Internal::~Buffer_Internal()
{
}

void Buffer_Internal::FetchBufferInfo()
{
    assert(0 < mPacketQty);

    unsigned int lSize_byte = sizeof(OpenNet_BufferHeader) + (sizeof(OpenNet_PacketInfo) * mPacketQty);
    assert(sizeof(mCache) > lSize_byte);

    Read(0, mCache, lSize_byte);
    Wait();
    assert(sizeof(OpenNet_BufferHeader) == mCache_Header->mPacketInfoOffset_byte);

    mCache_Info = reinterpret_cast<const OpenNet_PacketInfo *>(mCache + mCache_Header->mPacketInfoOffset_byte);
}

void Buffer_Internal::ResetMarkerValue()
{
    mMarkerValue = 0;
}

// ===== OpenNet::Buffer ====================================================

unsigned int Buffer_Internal::GetPacketCount() const
{
    assert(0 < mPacketQty);

    return mPacketQty;
}

unsigned int Buffer_Internal::GetPacketDestination(unsigned int aIndex) const
{
    assert(NULL != mCache_Info);
    assert(   0 <  mPacketQty );

    if (mPacketQty <= aIndex)
    {
        return 0;
    }

    return mCache_Info[aIndex].mSendTo & ( ~ ( OPEN_NET_PACKET_EVENT | OPEN_NET_PACKET_PROCESSED ) );
}

unsigned int Buffer_Internal::GetPacketEvent(unsigned int aFrom) const
{
    for (unsigned int i = aFrom; i < mPacketQty; i++)
    {
        if (0 != (OPEN_NET_PACKET_EVENT & mCache_Info[i].mSendTo))
        {
            return i;
        }
    }

    return 0xffffffff;
}

unsigned int Buffer_Internal::GetPacketSize(unsigned int aIndex) const
{
    assert(NULL != mCache_Info);
    assert(0    <  mPacketQty );

    if (mPacketQty <= aIndex)
    {
        return 0;
    }

    return mCache_Info[aIndex].mSize_byte;
}

OpenNet::Status Buffer_Internal::ClearEvent()
{
    try
    {
        Write(offsetof(OpenNet_BufferHeader, mEvents), &EVENT_CLEAR, sizeof(EVENT_CLEAR));
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Buffer_Internal::Display(FILE * aOut) const
{
    assert(NULL != mCache_Header);
    assert(NULL != mCache_Info  );

    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Buffer :\n");
    fprintf(aOut, "    Events              = 0x%08x\n"  , mCache_Header->mEvents               );
    fprintf(aOut, "    Packet Info. Offset = %u bytes\n", mCache_Header->mPacketInfoOffset_byte);
    fprintf(aOut, "    Packet Qty          = %u\n"      , mCache_Header->mPacketQty            );
    fprintf(aOut, "    Packet Size         = %u bytes\n", mCache_Header->mPacketSize_byte      );

    fprintf(aOut, "     #   Offset    Send To     Size\n");
    fprintf(aOut, "    ===  (bytes)  ==========  (bytes)\n");

    for (unsigned int i = 0; i < mCache_Header->mPacketQty; i ++)
    {
        fprintf(aOut, "    %3u  %7u  0x%08x  %7u\n", i, mCache_Info[i].mOffset_byte, mCache_Info[i].mSendTo, mCache_Info[i].mSize_byte);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Buffer_Internal::ReadPacket(unsigned int aIndex, void * aOut, unsigned int aOutSize_byte)
{
    assert(0 < mPacketQty);

    if (mPacketQty <= aIndex)
    {
        return OpenNet::STATUS_INVALID_INDEX;
    }

    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aOutSize_byte)
    {
        return OpenNet::STATUS_BUFFER_TOO_SMALL;
    }

    try
    {
        assert(NULL != mCache_Info);

        unsigned int lSize_byte = mCache_Info[aIndex].mSize_byte;
        if (0 < lSize_byte)
        {
            if (aOutSize_byte < lSize_byte)
            {
                lSize_byte = aOutSize_byte;
            }

            Read(mCache_Info[aIndex].mOffset_byte, aOut, lSize_byte);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Buffer_Internal::Wait()
{
    try
    {
        Wait_Internal();
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}
