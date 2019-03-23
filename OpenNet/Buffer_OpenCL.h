
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet ============================================================
#include "Buffer_Internal.h"
#include "Event_OpenCL.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_OpenCL : public Buffer_Internal
{

public:

    Buffer_OpenCL(bool aProfiling, cl_mem aMem, cl_command_queue aCommandQueue, unsigned int aPacketQty);

    // ===== Buffer_Internal ================================================
    ~Buffer_OpenCL();

    Event_OpenCL mEvent_OpenCL;
    cl_mem       mMem         ;

protected:

    // ===== Buffer_Internal ================================================

    virtual void Read (unsigned int aOffset_byte,       void * aOut, unsigned int aOutSize_byte);
    virtual void Write(unsigned int aOffset_byte, const void * aIn , unsigned int aInSize_byte );

    virtual void Wait_Internal();


private:

    void ReleaseEvent();

    cl_command_queue mCommandQueue;
    cl_event         mEvent       ;

};
