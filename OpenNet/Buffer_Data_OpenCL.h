
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Buffer_Data.h"

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data_OpenCL : public Buffer_Data
{

public:

    Buffer_Data_OpenCL(cl_mem aMem, unsigned int aPacketQty);

    // ===== Buffer_Data ====================================================

    virtual ~Buffer_Data_OpenCL();

    cl_event mEvent;
    cl_mem   mMem  ;

};
