
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Queue.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    NTSTATUS Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter);
}
