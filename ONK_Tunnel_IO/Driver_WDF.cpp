
// Author     KMS - Martin Dubois, ing
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Driver_WDF.cpp

// TEST COVERAGE  2019-03-19  KMS - Martin Dubois, ing.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== ONK_Pro1000 ========================================================
#include "Device_WDF.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_DRIVER_DEVICE_ADD DeviceAdd;
}

// Entry point
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    DRIVER_INITIALIZE DriverEntry;
}


// NOT TESTED  Driver.LoadUnload.ErrorHandling
//             WdfDriverCreate fail<br>
#pragma alloc_text (INIT, DriverEntry)

NTSTATUS DriverEntry(PDRIVER_OBJECT aDriverObject, PUNICODE_STRING aRegistryPath)
{
    DbgPrintEx(DEBUG_ID, DEBUG_INFO, PREFIX "OpenNet - ONK_Tunnel_IO"               DEBUG_EOL);
    DbgPrintEx(DEBUG_ID, DEBUG_INFO, PREFIX "Version " VERSION_STR " " VERSION_TYPE DEBUG_EOL);
    DbgPrintEx(DEBUG_ID, DEBUG_INFO, PREFIX "Compiled at " __TIME__ " on " __DATE__ DEBUG_EOL);
    DbgPrintEx(DEBUG_ID, DEBUG_INFO, PREFIX "Purchased by " VERSION_CLIENT          DEBUG_EOL);

    ASSERT(NULL != aDriverObject);
    ASSERT(NULL != aRegistryPath);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT(&lAttributes);

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

#pragma alloc_text (PAGE, DeviceAdd)

NTSTATUS DeviceAdd(WDFDRIVER aDriver, PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDriver);
    ASSERT(NULL != aDeviceInit);

    UNREFERENCED_PARAMETER(aDriver);

    PAGED_CODE();

    return Device_Create(aDeviceInit);
}
