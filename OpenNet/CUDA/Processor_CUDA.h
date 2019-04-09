
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Processor_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== VIDIA ==============================================================
#include <cuda.h>
#include <nvrtc.h>

// ===== OpenNet/CUDA =======================================================
#include "../Internal/Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_CUDA : public Processor_Internal
{

public:

    Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog );

    Buffer_Internal * Buffer_Allocate( bool aProfiling, unsigned int aPacketSize_byte, OpenNetK::Buffer * aBuffer);

    CUmodule Module_Create(OpenNet::Kernel * aKernel, unsigned int aAdapterNo );

    void SetContext();

    // ====== Processor_Internal ============================================

    virtual Thread_Functions * Thread_Get();

    // ===== OpenNet::Processor =============================================

    virtual ~Processor_CUDA();

    virtual void          * GetContext ();
    virtual void          * GetDevice  ();

protected:

    // ====== Processor_Internal ============================================

    virtual OpenNet::UserBuffer * AllocateUserBuffer_Internal( unsigned int aSize_byte );

private:

    void InitInfo();

    nvrtcProgram Program_CreateAndCompile( OpenNet::Kernel * aKernel, unsigned int aAdapterNo );

    CUcontext mContext;
    CUdevice  mDevice ;

};
