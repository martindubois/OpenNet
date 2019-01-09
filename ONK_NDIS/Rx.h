
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Rx.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_NDIS ===========================================================
#include "VirtualHardware.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void Rx_Create(NETRXQUEUE_INIT * aQueueInit, VirtualHardware * aHardware);
