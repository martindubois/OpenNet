
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Queue.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    void Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter);
}
