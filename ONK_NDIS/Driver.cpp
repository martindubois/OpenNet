
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Driver.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// =====  ===================================================================
#include <TraceLoggingProvider.h>

// ===== ONK_NDIS ===========================================================
#include "ControlDevice.h"
#include "NdisDevice.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static DECLARE_CONST_UNICODE_STRING(SDDL, L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;AU)");

// {C95DF9B7-A9C3-4046-9829-30DB9A97CC1B}
TRACELOGGING_DEFINE_PROVIDER( TRACE_PROVIDER, "ONK_NDIS.Trace.Provider", ( 0xc95df9b7, 0xa9c3, 0x4046, 0x98, 0x29, 0x30, 0xdb, 0x9a, 0x97, 0xcc, 0x1b ) );

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

    NTSTATUS lStatus = TraceLoggingRegister(TRACE_PROVIDER);
    ASSERT(STATUS_SUCCESS == lStatus);

    WDF_DRIVER_CONFIG lConfig;

    WDF_DRIVER_CONFIG_INIT(&lConfig, DeviceAdd);

    lConfig.DriverPoolTag   = TAG   ;
    lConfig.EvtDriverUnload = Unload;

    lStatus = WdfDriverCreate(aDrvObj, aRegPath, WDF_NO_OBJECT_ATTRIBUTES, &lConfig, NULL);
    ASSERT(STATUS_SUCCESS == lStatus);
    (void)(lStatus);

    return STATUS_SUCCESS;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS DeviceAdd(WDFDRIVER aDriver, PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDriver    );
    ASSERT(NULL != aDeviceInit);

    NTSTATUS lResult = NdisDevice_Create(aDeviceInit);
    if (STATUS_SUCCESS == lResult)
    {
        PWDFDEVICE_INIT lControlDeviceInit = WdfControlDeviceInitAllocate(aDriver, &SDDL);
        ASSERT(NULL != lControlDeviceInit);

        ControlDevice_Create(lControlDeviceInit);
    }

    return lResult;
}

VOID Unload(WDFDRIVER aDriver)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDriver);

    UNREFERENCED_PARAMETER(aDriver);

    // TODO  ONK_NDIS
    //       Normal (Feature)
}
