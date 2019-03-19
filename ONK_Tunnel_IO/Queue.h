
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Queue.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_Tunnel_IO ======================================================
class VirtualHardware;

// Functions
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    NTSTATUS Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter, VirtualHardware * aHardware);
}
