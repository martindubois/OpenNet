
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet ============================================================
#include "Event_OpenCL.h"

#include "Buffer_Data.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data_OpenCL : public Buffer_Data
{

public:

    Buffer_Data_OpenCL( bool aProfiling, cl_mem aMem, unsigned int aPacketQty);

    // ===== Buffer_Data ====================================================

    virtual ~Buffer_Data_OpenCL();

    Event_OpenCL mEvent_OpenCL;

    cl_mem   mMem  ;

};
