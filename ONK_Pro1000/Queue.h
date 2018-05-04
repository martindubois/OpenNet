
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Queue.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL Queue_IoDeviceControl;
    EVT_WDF_IO_QUEUE_IO_STOP           Queue_IoStop;
}
