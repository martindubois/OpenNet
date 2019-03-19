
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Device_WDF.cpp

// TEST COVERAGE  2019-03-29  KMS - Martin Dubois, ing.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_WDF.h>
#include <OpenNetK/Hardware_WDF.h>
#include <OpenNetK/Interface.h>

// ===== ONK_Tunnel_IO ======================================================
#include "Queue.h"
#include "VirtualHardware.h"

#include "Device_WDF.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter      mAdapter     ;
    VirtualHardware        mHardware    ;
    OpenNetK::Adapter_WDF  mAdapter_WDF ;
    OpenNetK::Hardware_WDF mHardware_WDF;
}
DeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DeviceContext, GetDeviceContext)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static NTSTATUS Init(DeviceContext * aThis, WDFDEVICE aDevice);

static void DeviceInit_Init(PWDFDEVICE_INIT aDeviceInit);

// ===== Entry points =======================================================
extern "C"
{
    static EVT_WDF_DEVICE_D0_ENTRY         D0Entry          ;
    static EVT_WDF_DEVICE_D0_EXIT          D0Exit           ;
    static EVT_WDF_FILE_CLEANUP            FileCleanup      ;
    static EVT_WDF_IO_IN_CALLER_CONTEXT    IoInCallerContext;
    static EVT_WDF_DEVICE_PREPARE_HARDWARE PrepareHardware  ;
    static EVT_WDF_DEVICE_RELEASE_HARDWARE ReleaseHardware  ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// aDeviceInit [---;RW-]
//
// Return  STATUS_SUCCESS
//         See WdfDeviceCreate
//
// Thread  PnP

// NOT TESTED  Driver.LoadUnload.ErrorHandling
//             WdfDeviceCreate fail<br>
//             WdfDeviceCreateDeviceInterface fail<br>

#pragma alloc_text (PAGE, Device_Create)

NTSTATUS Device_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    PAGED_CODE();

    DeviceInit_Init(aDeviceInit);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, DeviceContext);

    WDFDEVICE lDevice;

    NTSTATUS lStatus = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
    if (STATUS_SUCCESS != lStatus)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDeviceCreate( , ,  ) failed - 0x%08x" DEBUG_EOL, lStatus);
        return lStatus;
    }

    DeviceContext * lThis = GetDeviceContext(lDevice);
    ASSERT(NULL != lThis);

    lStatus = Init(lThis, lDevice);
    if (STATUS_SUCCESS != lStatus)
    {
        return lStatus;
    }

    lStatus = WdfDeviceCreateDeviceInterface(lDevice, &OPEN_NET_DRIVER_INTERFACE, NULL);
    if (STATUS_SUCCESS != lStatus)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDeviceCreateDeviceInterface( , ,  ) failed - 0x%08x" DEBUG_EOL, lStatus);
        return lStatus;
    }

    lStatus = Queue_Create(lDevice, &lThis->mAdapter_WDF, &lThis->mHardware);
    if (STATUS_SUCCESS != lStatus)
    {
        return lStatus;
    }

    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ " - OK" DEBUG_EOL);

    return STATUS_SUCCESS;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aThis   [---;RW-]
// aDevice [-K-;RW-]
//
// Return  STATUS_SUCCESS
//         See Hardware_WDF::Init
//
// Thread  PnP
NTSTATUS Init(DeviceContext * aThis, WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aThis  );
    ASSERT(NULL != aDevice);

    new (&aThis->mHardware) VirtualHardware();

    NTSTATUS lStatus = aThis->mHardware_WDF.Init(aDevice, &aThis->mHardware);
    if (STATUS_SUCCESS != lStatus)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - Hardware_WDF::Init( , ,  ) failed - 0x%08x" DEBUG_EOL, lStatus);
        return lStatus;
    }

    aThis->mAdapter_WDF.Init(&aThis->mAdapter, aDevice, &aThis->mHardware_WDF);

    aThis->mAdapter.SetHardware(&aThis->mHardware);

    aThis->mHardware.SetAdapter(&aThis->mAdapter);

    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ " - OK" DEBUG_EOL);

    return STATUS_SUCCESS;
}

// aDeviceInit [---;RW-]
//
// Thread  PnP
void DeviceInit_Init(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    WDF_FILEOBJECT_CONFIG lFileObjectConfig;

    WDF_FILEOBJECT_CONFIG_INIT(&lFileObjectConfig, NULL, NULL, FileCleanup);

    WdfDeviceInitSetFileObjectConfig(aDeviceInit, &lFileObjectConfig, WDF_NO_OBJECT_ATTRIBUTES);

    WdfDeviceInitSetIoInCallerContextCallback(aDeviceInit, IoInCallerContext);

    WDF_PNPPOWER_EVENT_CALLBACKS lPnpPowerEventCallbacks;

    WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&lPnpPowerEventCallbacks);

    lPnpPowerEventCallbacks.EvtDeviceD0Entry = D0Entry;
    lPnpPowerEventCallbacks.EvtDeviceD0Exit = D0Exit;
    lPnpPowerEventCallbacks.EvtDevicePrepareHardware = PrepareHardware;
    lPnpPowerEventCallbacks.EvtDeviceReleaseHardware = ReleaseHardware;

    WdfDeviceInitSetPnpPowerEventCallbacks(aDeviceInit, &lPnpPowerEventCallbacks);
}

// ===== Entry points =======================================================

NTSTATUS D0Entry(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aPreviousState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , 0x%08x )" DEBUG_EOL, aPreviousState);

    ASSERT(NULL                  != aDevice       );
    ASSERT(WdfPowerDeviceD0      != aPreviousState);
    ASSERT(WdfPowerDeviceInvalid != aPreviousState);
    ASSERT(WdfPowerDeviceMaximum >  aPreviousState);

    DeviceContext * lThis = GetDeviceContext(aDevice);
    ASSERT(NULL != lThis);

    // Hardware_WDF::D0Entry ==> Hardware_WDF::D0Exist  See D0Exit
    return lThis->mHardware_WDF.D0Entry(aPreviousState);
}

NTSTATUS D0Exit(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aTargetState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , 0x%08x )" DEBUG_EOL, aTargetState);

    ASSERT(NULL                  != aDevice     );
    ASSERT(WdfPowerDeviceD0      != aTargetState);
    ASSERT(WdfPowerDeviceInvalid != aTargetState);
    ASSERT(WdfPowerDeviceMaximum >  aTargetState);

    DeviceContext * lThis = GetDeviceContext(aDevice);
    ASSERT(NULL != lThis);

    // Hardware_WDF::D0Entry ==> Hardware_WDF::D0Exist  See D0Entry
    return lThis->mHardware_WDF.D0Exit(aTargetState);
}

void FileCleanup(WDFFILEOBJECT aFileObject)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aFileObject);

    WDFDEVICE lDevice = WdfFileObjectGetDevice(aFileObject);
    ASSERT(NULL != lDevice);

    DeviceContext * lThis = GetDeviceContext(lDevice);
    ASSERT(NULL != lThis);

    lThis->mAdapter_WDF.FileCleanup(aFileObject);
}

void IoInCallerContext(WDFDEVICE aDevice, WDFREQUEST aRequest)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice );
    ASSERT(NULL != aRequest);

    DeviceContext * lThis = GetDeviceContext(aDevice);

    lThis->mAdapter_WDF.IoInCallerContext(aRequest);
}

NTSTATUS PrepareHardware(WDFDEVICE aDevice, WDFCMRESLIST aRaw, WDFCMRESLIST aTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice    );
    ASSERT(NULL != aRaw       );
    ASSERT(NULL != aTranslated);

    DeviceContext * lThis = GetDeviceContext(aDevice);
    ASSERT(NULL != lThis);

    // Hardware_WDF::PrepareHardware ==> Hardware_WDF::ReleaseHardware  See D0Exit
    return lThis->mHardware_WDF.PrepareHardware(aRaw, aTranslated);
}

NTSTATUS ReleaseHardware(WDFDEVICE aDevice, WDFCMRESLIST aTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice    );
    ASSERT(NULL != aTranslated);

    DeviceContext * lThis = GetDeviceContext(aDevice);
    ASSERT(NULL != lThis);

    // Hardware_WDF::PrepareHardware ==> Hardware_WDF::ReleaseHardware  See D0Exit
    return lThis->mHardware_WDF.ReleaseHardware(aTranslated);
}
