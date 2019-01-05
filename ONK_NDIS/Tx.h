#pragma once

// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Tx.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_NDIS ===========================================================
#include "VirtualHardware.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

extern  void Tx_Create(NETTXQUEUE_INIT * aQueueInit, VirtualHardware * aHardware);
