
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

#ifdef _KMS_WINDOWS_
    // ===== OpenCL =========================================================
    #include <CL/opencl.h>
#endif

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data
{

public:

    #ifdef _KMS_WINDOWS_
        Buffer_Data(cl_mem aMem, unsigned int aPacketQty);
    #endif

    ~Buffer_Data();

    uint32_t     GetMarkerValue();
    unsigned int GetPacketQty  () const;

    void ResetMarkerValue();

    #ifdef _KMS_WINDOWS_
        cl_event mEvent;
        cl_mem   mMem  ;
    #endif

private:

    uint32_t     mMarkerValue;
    unsigned int mPacketQty  ;

};

typedef std::vector<Buffer_Data *> Buffer_Data_Vector;
