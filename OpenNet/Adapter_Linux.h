
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Linux.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Linux : public Adapter_Internal
{

public:

    Adapter_Linux(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    void Buffers_Allocate( Buffer_Data_Vector * aBuffers );

    // ===== OpenNet::Adapter ==============================================

    virtual ~Adapter_Linux();

protected:

    // ===== Adapter_Internal ===============================================

    virtual void ResetInputFilter_Internal();
    virtual void SetInputFilter_Internal  (OpenNet::Kernel * aKernel);

    virtual Thread * Thread_Prepare_Internal(OpenNet::Kernel * aKernel);

private:

    Buffer_Data * Buffer_Allocate();

    CUmodule mModule;

};
