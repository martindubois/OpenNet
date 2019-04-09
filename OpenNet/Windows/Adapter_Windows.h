
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Windows/Adapter_Windows.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet/Windows ====================================================
#include "../Internal/Adapter_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Windows : public Adapter_Internal
{

public:

    Adapter_Windows(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    void Buffers_Allocate( bool aProfiling, cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Internal_Vector * aBuffers);

    // ===== OpenNet::Adapter ===============================================

    virtual ~Adapter_Windows();

protected:

    // ===== Adapter_Internal ===============================================

    virtual void ResetInputFilter_Internal();
    virtual void SetInputFilter_Internal  (OpenNet::Kernel * aKernel);

    virtual void Stop_Internal();

    virtual Thread * Thread_Prepare_Internal(OpenNet::Kernel * aKernel);

private:

    Buffer_Internal * Buffer_Allocate(bool aProfiling, cl_command_queue aCommandQueue, cl_kernel aKernel);

    cl_program mProgram;

};
