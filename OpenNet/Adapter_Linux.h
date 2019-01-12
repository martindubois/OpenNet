
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

    virtual Thread * Thread_Prepare();

    // ===== OpenNet::Adapter ==============================================

    virtual ~Adapter_Linux();

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

};
