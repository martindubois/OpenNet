
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Linux.h

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

    Adapter_Linux(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    // ===== Adapter_Internal ===============================================

    virtual void Connect(IoCtl_Connect_In * aConnect);

    // ===== OpenNet::Adapter ==============================================

    virtual ~Adapter_Linux();

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

protected:

    // ===== Adapter_Internal ===============================================

    virtual OpenNet::Status ResetInputFilter_Internal();
    virtual void            SetInputFilter_Internal  (OpenNet::Kernel * aKernel);

    virtual Thread * Thread_Prepare_Internal(OpenNet::Kernel * aKernel);


};
