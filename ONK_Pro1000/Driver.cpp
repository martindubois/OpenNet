
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_Pro1000/Driver.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== ONK_Pro1000 ========================================================
#include "Device.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_OBJECT_CONTEXT_CLEANUP Cleanup  ;
    static EVT_WDF_DRIVER_DEVICE_ADD      DeviceAdd;
}

// Entry point
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    DRIVER_INITIALIZE DriverEntry;
}

// Thread  PnP

// NOT TESTED  ONK_Pro1000.Driver.ErrorHandling
//             WdfDriverCreate fail

#pragma alloc_text (INIT, DriverEntry)

NTSTATUS DriverEntry(PDRIVER_OBJECT aDriverObject, PUNICODE_STRING aRegistryPath)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDriverObject);
    ASSERT(NULL != aRegistryPath);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT(&lAttributes);
    lAttributes.EvtCleanupCallback = Cleanup;

    WDF_DRIVER_CONFIG lConfig;

    WDF_DRIVER_CONFIG_INIT(&lConfig, DeviceAdd);

    NTSTATUS lResult = WdfDriverCreate(aDriverObject, aRegistryPath, &lAttributes, &lConfig, WDF_NO_HANDLE);
    if (STATUS_SUCCESS != lResult)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDriverCreate( , , , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

// Thread  PnP

#pragma alloc_text (PAGE, Cleanup)

VOID Cleanup(WDFOBJECT aDriverObject)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDriverObject);

    UNREFERENCED_PARAMETER(aDriverObject);

    PAGED_CODE ();
}

// Thread  PnP

// NOT TESTED  ONK_Pro1000.Driver.ErrorHandling
//             Device_Create fail

#pragma alloc_text (PAGE, DeviceAdd)

NTSTATUS DeviceAdd(WDFDRIVER aDriver, PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDriver    );
    ASSERT(NULL != aDeviceInit);

    UNREFERENCED_PARAMETER(aDriver);

    PAGED_CODE();

    NTSTATUS lResult = Device_Create(aDeviceInit);
    if (STATUS_SUCCESS != lResult)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, __FUNCTION__ " - Device_Create(  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}
