
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved
// Product    OpenNet
// File       ONK_Tunnel_IO/Device_WDF.h

#pragma once

// Function
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    NTSTATUS Device_Create(PWDFDEVICE_INIT DeviceInit);
}
