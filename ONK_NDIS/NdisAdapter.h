
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/NdisAdapter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_NDIS ===========================================================
#include "VirtualHardware.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

extern NTSTATUS NdisAdapter_Create(WDFDEVICE aDevice, void ** aAdapter, VirtualHardware * aHardware);
