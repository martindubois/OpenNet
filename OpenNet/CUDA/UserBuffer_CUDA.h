
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/UserBuffer_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet/CUDA =======================================================
#include "../Internal/UserBuffer_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class UserBuffer_CUDA : public UserBuffer_Internal
{

public:

    UserBuffer_CUDA( unsigned int aSize_byte );

    CUdeviceptr mMem_DA;

protected:

    // ===== UserBuffer_Internal ============================================
    virtual void Clear_Internal();
    virtual void Read_Internal (unsigned int aOffset_byte,       void * aOut, unsigned int aSize_byte);
    virtual void Write_Internal(unsigned int aOffset_byte, const void * aIn , unsigned int aSize_byte);

    // ===== OpenNet::UserBuffer ============================================
    virtual ~UserBuffer_CUDA();

private:

};
