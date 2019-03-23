
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Includes ===========================================================
#include <OpenNet/Buffer.h>
#include <OpenNetK/Types.h>

// ===== OpenNet ============================================================
class Event;

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Internal : public OpenNet::Buffer
{

public:

    virtual ~Buffer_Internal();

    Event      * GetEvent      ();
    uint32_t     GetMarkerValue();
    unsigned int GetPacketQty  () const;

    void FetchBufferInfo();

    void ResetMarkerValue();

    // ===== OpenNet::Buffer ================================================
    virtual unsigned int    GetPacketCount      () const;
    virtual uint32_t        GetPacketDestination(unsigned int aIndex) const;
    virtual unsigned int    GetPacketEvent      (unsigned int aIndex) const;
    virtual unsigned int    GetPacketSize       (unsigned int aIndex) const;
    virtual OpenNet::Status ClearEvent          ();
    virtual OpenNet::Status Display             (FILE * aOut) const;
    virtual OpenNet::Status ReadPacket          (unsigned int aIndex, void * aOut, unsigned int aOutSize_byte);
    virtual OpenNet::Status Wait                ();

protected:

    Buffer_Internal(unsigned int aPacketQty, Event * aEvent);

    virtual void Read (unsigned int aOffset_byte,       void * aOut, unsigned int aOutSize_byte) = 0;
    virtual void Write(unsigned int aOffset_byte, const void * aIn , unsigned int aInSize_byte ) = 0;

    virtual void Wait_Internal() = 0;

private:

    uint8_t                      mCache[8192] ;
    const OpenNet_BufferHeader * mCache_Header;
    const OpenNet_PacketInfo   * mCache_Info  ;
    Event                      * mEvent       ;
    uint32_t                     mMarkerValue ;
    unsigned int                 mPacketQty   ;

};

typedef std::vector<Buffer_Internal *> Buffer_Internal_Vector;

// Public
/////////////////////////////////////////////////////////////////////////////

// Return  This method returns the event associated to the buffer

// CRITICAL PATH  Processing
//                1 / buffer in KERNEL mode
inline Event * Buffer_Internal::GetEvent()
{
    return mEvent;
}

// Return  The next value to wait for

// CRITICAL PATH  Processing
//                1 / budder in KERNEL mode (OpenCL only)
inline uint32_t Buffer_Internal::GetMarkerValue()
{
    mMarkerValue++;

    return mMarkerValue;
}

// Return  The number of packet the buffer contains

// CRITICAL PATH  Processing
//                1 / buffer or 1 / processing iteration
inline unsigned int Buffer_Internal::GetPacketQty() const
{
    return mPacketQty;
}
