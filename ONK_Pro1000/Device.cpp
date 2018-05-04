
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Interface.h>

// ===== ONK_Pro1000 ========================================================
#include "Queue.h"

#include "Device.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    ULONG PrivateDeviceData;  // just a placeholder
}
DEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, DeviceGetContext)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    static NTSTATUS InitQueue(WDFDEVICE aDevice);
}

// Functions
/////////////////////////////////////////////////////////////////////////////

#pragma alloc_text (PAGE, Device_Create)

NTSTATUS Device_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    PAGED_CODE();

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, DEVICE_CONTEXT);

    WDFDEVICE lDevice;

    NTSTATUS lResult = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
    if ( STATUS_SUCCESS == lResult )
    {
        ASSERT(NULL != lDevice);

        DEVICE_CONTEXT * lThis = DeviceGetContext(lDevice);
        ASSERT(NULL != lThis);

        // TODO Dev
        (void)(lThis);

        lResult = WdfDeviceCreateDeviceInterface(lDevice, &OPEN_NET_DRIVER_INTERFACE, NULL);
        if (STATUS_SUCCESS == lResult)
        {
            lResult = InitQueue(lDevice);
        }
        else
        {
            DbgPrintEx(DEBUG_ID, DEBUG_ERROR, __FUNCTION__ " - WdfDeviceCreateDeviceInterface( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
        }
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, __FUNCTION__ " - WdfDeviceCreate( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

#pragma alloc_text (PAGE, InitQueue)

NTSTATUS InitQueue(WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDevice);

    PAGED_CODE();

    WDF_IO_QUEUE_CONFIG lConfig;

    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&lConfig, WdfIoQueueDispatchSequential);

    lConfig.EvtIoDeviceControl = Queue_IoDeviceControl;
    lConfig.EvtIoStop          = Queue_IoStop         ;

    WDFQUEUE lQueue;

    NTSTATUS lResult = WdfIoQueueCreate(aDevice, &lConfig, WDF_NO_OBJECT_ATTRIBUTES, &lQueue);
    if (STATUS_SUCCESS == lResult)
    {
        ASSERT(NULL != lQueue);
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfIoQueueCreate( , , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}
