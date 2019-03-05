
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== VIDIA ==============================================================
#include <cuda.h>
#include <nvrtc.h>

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_CUDA : public Processor_Internal
{

public:

    Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog );

    Buffer_Data * Buffer_Allocate( bool aProfiling, unsigned int aPacketSize_byte, OpenNetK::Buffer * aBuffer);

    CUmodule Module_Create(OpenNet::Kernel * aKernel);

    void SetContext();

    // ====== Processor_Internal ============================================

    virtual Thread_Functions * Thread_Get();

    // ===== OpenNet::Processor =============================================

    virtual ~Processor_CUDA();

    virtual void          * GetContext ();
    virtual void          * GetDevice  ();

private:

    void InitInfo();

    nvrtcProgram Program_CreateAndCompile( OpenNet::Kernel * aKernel );

    CUcontext mContext;
    CUdevice  mDevice ;

};
