
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/UserBuffer_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet ============================================================
#include "UserBuffer_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class UserBuffer_OpenCL : public UserBuffer_Internal
{

public:

    UserBuffer_OpenCL(unsigned int aSize_byte, cl_context aContext, cl_command_queue aCommandQueue);

    cl_mem mMem;

protected:

    // ===== UserBuffer_Internal ============================================
    virtual void Clear_Internal();
    virtual void Read_Internal (unsigned int aOffset_byte,       void * aOut, unsigned int aSize_byte);
    virtual void Write_Internal(unsigned int aOffset_byte, const void * aIn , unsigned int aSize_byte);

    // ===== OpenNet::UserBuffer ============================================
    virtual ~UserBuffer_OpenCL();

private:

    cl_command_queue mCommandQueue;

};
