
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Buffer_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data
{

public:

    Buffer_Data(cl_mem aMem, unsigned int aPacketQty);

    ~Buffer_Data();

    uint32_t     GetMarkerValue();
    unsigned int GetPacketQty  () const;

    void ResetMarkerValue();

    cl_event mEvent;
    cl_mem   mMem  ;

private:

    uint32_t     mMarkerValue;
    unsigned int mPacketQty  ;

};

typedef std::vector<Buffer_Data *> Buffer_Data_Vector;
