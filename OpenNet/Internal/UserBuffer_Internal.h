
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/UserBuffer_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/UserBuffer.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class UserBuffer_Internal : public OpenNet::UserBuffer
{

public:

    // ===== OpenNet::UserBuffer ============================================
    virtual OpenNet::Status Clear();
    virtual OpenNet::Status Read (unsigned int aOffset_byte,       void * aOut, unsigned int aSize_byte);
    virtual OpenNet::Status Write(unsigned int aOffset_byte, const void * aIn , unsigned int aSize_byte);

protected:

    UserBuffer_Internal(unsigned int aSize_byte);

    virtual void Clear_Internal() = 0;
    virtual void Read_Internal (unsigned int aOffset_byte,       void * aOut, unsigned int aSize_byte) = 0;
    virtual void Write_Internal(unsigned int aOffset_byte, const void * aIn , unsigned int aSize_byte) = 0;

    unsigned int mSize_byte;

};
