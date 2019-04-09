
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Buffer_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet/CUDA ============================================================

#include "../Internal/Buffer_Internal.h"

#include "Event_CUDA.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_CUDA : public Buffer_Internal
{

public:

    Buffer_CUDA(bool aProfiling, CUcontext aContext, CUdeviceptr aMem_DA, unsigned int aPacketQty );

    // ===== Buffer_Internal ================================================
    ~Buffer_CUDA();

    virtual void FetchBufferInfo();

    Event_CUDA  mEvent_CUDA;
    CUcontext   mContext   ;
    CUdeviceptr mMemory_DA ;

protected:

    // ===== Buffer_Internal ================================================

    virtual void Read (unsigned int aOffset_byte,       void * aOut, unsigned int aOutSize_byte);
    virtual void Write(unsigned int aOffset_byte, const void * aIn , unsigned int aInSize_byte );

    virtual void Wait_Internal();

};
