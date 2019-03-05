
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== OpenNet ============================================================
class Event;

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data
{

public:

    Buffer_Data(unsigned int aPacketQty, Event * aEvent);

    virtual ~Buffer_Data();

    Event      * GetEvent      ();
    uint32_t     GetMarkerValue();
    unsigned int GetPacketQty  () const;

    void ResetMarkerValue();


private:

    Event      * mEvent      ;
    uint32_t     mMarkerValue;
    unsigned int mPacketQty  ;

};

typedef std::vector<Buffer_Data *> Buffer_Data_Vector;

// Public
/////////////////////////////////////////////////////////////////////////////

// Return  This method returns the event associated to the buffer

// CRITICAL PATH  Processing
//                1 / buffer in KERNEL mode
inline Event * Buffer_Data::GetEvent()
{
    return mEvent;
}

// Return  The next value to wait for

// CRITICAL PATH  Processing
//                1 / budder in KERNEL mode (OpenCL only)
inline uint32_t Buffer_Data::GetMarkerValue()
{
    mMarkerValue++;

    return mMarkerValue;
}

// Return  The number of packet the buffer contains

// CRITICAL PATH  Processing
//                1 / buffer or 1 / processing iteration
inline unsigned int Buffer_Data::GetPacketQty() const
{
    return mPacketQty;
}
