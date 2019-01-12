
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Windows.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Linux : public Adapter_Internal
{

public:

    Adapter_Internal(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    void Buffers_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Data_Vector * aBuffers);

    // ===== Adapter_Internal ===============================================
    
    virtual void Connect(IoCtl_Connect_In * aConnect);

    virtual Thread * Thread_Prepare();

    // ===== OpenNet::Adapter ===============================================

    virtual ~Adapter_Windows();

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

private:

    Buffer_Data * Buffer_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel);

    cl_program mProgram;

};
