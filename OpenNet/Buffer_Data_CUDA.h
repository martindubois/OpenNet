
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_Data_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet ============================================================
#include "Buffer_Data.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data_CUDA : public Buffer_Data
{

public:

    Buffer_Data_CUDA( CUdeviceptr aMem_DA, unsigned int aPacketQty );

    // ===== Buffer_Data ====================================================

    virtual ~Buffer_Data_CUDA();

    CUdeviceptr mMemory_DA;

};
