
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Driver.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== ONK_NDIS ===========================================================
#include "ControlDevice.h"
#include "NdisDevice.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static DECLARE_CONST_UNICODE_STRING(SDDL, L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;AU)");

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    static EVT_WDF_DRIVER_DEVICE_ADD DeviceAdd;
    static EVT_WDF_DRIVER_UNLOAD     Unload   ;
}

// Entry point / Point d'entre
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    DRIVER_INITIALIZE DriverEntry;
}

#pragma alloc_text (INIT, DriverEntry)

NTSTATUS DriverEntry(PDRIVER_OBJECT aDrvObj, PUNICODE_STRING aRegPath)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDrvObj );
    ASSERT(NULL != aRegPath);

    WDF_DRIVER_CONFIG lConfig;

    WDF_DRIVER_CONFIG_INIT(&lConfig, DeviceAdd);

    lConfig.DriverPoolTag   = TAG   ;
    lConfig.EvtDriverUnload = Unload;

    NTSTATUS lResult = WdfDriverCreate(aDrvObj, aRegPath, WDF_NO_OBJECT_ATTRIBUTES, &lConfig, NULL);
    if (STATUS_SUCCESS != lResult)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDriverCreate( , , , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS DeviceAdd(WDFDRIVER aDriver, PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDriver    );
    ASSERT(NULL != aDeviceInit);

    UNREFERENCED_PARAMETER(aDriver    );
    UNREFERENCED_PARAMETER(aDeviceInit);

    NTSTATUS lResult = NdisDevice_Create(aDeviceInit);
    if (STATUS_SUCCESS == lResult)
    {
        PWDFDEVICE_INIT lControlDeviceInit = WdfControlDeviceInitAllocate(aDriver, &SDDL);
        ASSERT(NULL != lControlDeviceInit);

        lResult = ControlDevice_Create(lControlDeviceInit);
    }

    return lResult;
}

VOID Unload(WDFDRIVER aDriver)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDriver);

    UNREFERENCED_PARAMETER(aDriver);

    // TODO Dev
}
