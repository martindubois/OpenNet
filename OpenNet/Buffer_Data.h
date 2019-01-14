
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data
{

public:

    Buffer_Data(unsigned int aPacketQty);

    virtual ~Buffer_Data();

    uint32_t     GetMarkerValue();
    unsigned int GetPacketQty  () const;

    void ResetMarkerValue();


private:

    uint32_t     mMarkerValue;
    unsigned int mPacketQty  ;

};

typedef std::vector<Buffer_Data *> Buffer_Data_Vector;
