
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Interface.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_WDF.h>
#include <OpenNetK/Hardware_WDF.h>

// ===== ONK_Pro1000 ========================================================
#include "Queue.h"
#include "Pro1000.h"

#include "Device.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter_WDF  mAdapter_WDF ;
    OpenNetK::Hardware_WDF mHardware_WDF;

    // ===== Zone 0 =========================================================
    WDFSPINLOCK       mZone0   ;
    OpenNetK::Adapter mAdapter ;
    Pro1000           mHardware;
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
    static EVT_WDF_FILE_CLOSE              FileClose        ;
    static EVT_WDF_DEVICE_FILE_CREATE      FileCreate       ;
    static EVT_WDF_IO_IN_CALLER_CONTEXT    IoInCallerContext;
    static EVT_WDF_DEVICE_PREPARE_HARDWARE PrepareHardware  ;
    static EVT_WDF_DEVICE_RELEASE_HARDWARE ReleaseHardware  ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

#pragma alloc_text (PAGE, Device_Create)

// Thread  PnP

// NOT TESTED  ONK_Pro1000.Device.ErrorHandling
//             WdfDeviceCreate fail<br>
//             WdfDeviceCreateDeviceInterface fail
NTSTATUS Device_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    PAGED_CODE();

    DeviceInit_Init(aDeviceInit);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, DeviceContext);

    WDFDEVICE lDevice;

    NTSTATUS lResult = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
    if ( STATUS_SUCCESS == lResult )
    {
        ASSERT(NULL != lDevice);

        DeviceContext * lThis = GetDeviceContext(lDevice);
        ASSERT(NULL != lThis);

        lResult = Init(lThis, lDevice);
        if (STATUS_SUCCESS == lResult)
        {
            lResult = WdfDeviceCreateDeviceInterface(lDevice, &OPEN_NET_DRIVER_INTERFACE, NULL);
            if (STATUS_SUCCESS == lResult)
            {
                lResult = Queue_Create(lDevice, &lThis->mAdapter_WDF);
            }
            else
            {
                DbgPrintEx(DEBUG_ID, DEBUG_ERROR, __FUNCTION__ " - WdfDeviceCreateDeviceInterface( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
            }
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

// aThis   [---;RW-]
// aDevice [-K-;RW-]
//
// Thread  PnP

// NOT TESTED  ONK_Pro1000.Device.ErrorHandling
//             Hardware_WDF.Init fail
NTSTATUS Init(DeviceContext * aThis, WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aThis);
    ASSERT(NULL != aDevice);

    WDF_OBJECT_ATTRIBUTES lAttr;

    WDF_OBJECT_ATTRIBUTES_INIT(&lAttr);

    lAttr.ParentObject = aDevice;

    NTSTATUS lResult = WdfSpinLockCreate(&lAttr, &aThis->mZone0);
    if (STATUS_SUCCESS == lResult)
    {
        new (&aThis->mHardware) Pro1000();

        lResult = aThis->mHardware_WDF.Init(aDevice, &aThis->mHardware, aThis->mZone0);
        if (STATUS_SUCCESS == lResult)
        {
            aThis->mAdapter_WDF.Init(&aThis->mAdapter, aDevice, &aThis->mHardware_WDF, aThis->mZone0);
            aThis->mAdapter.SetHardware(&aThis->mHardware);

            aThis->mHardware.SetAdapter(&aThis->mAdapter);
        }
        else
        {
            DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - Hardware_WDF::Init( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
        }
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfSpinLockCreate( ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Thread  PnP
void DeviceInit_Init(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    WDF_FILEOBJECT_CONFIG lFileObjectConfig;

    WDF_FILEOBJECT_CONFIG_INIT(&lFileObjectConfig, FileCreate, FileClose, FileCleanup);

    WdfDeviceInitSetFileObjectConfig(aDeviceInit, &lFileObjectConfig, WDF_NO_OBJECT_ATTRIBUTES);

    WdfDeviceInitSetIoInCallerContextCallback(aDeviceInit, IoInCallerContext);

    WDF_PNPPOWER_EVENT_CALLBACKS lPnpPowerEventCallbacks;

    WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&lPnpPowerEventCallbacks);

    lPnpPowerEventCallbacks.EvtDeviceD0Entry         = D0Entry        ;
    lPnpPowerEventCallbacks.EvtDeviceD0Exit          = D0Exit         ;
    lPnpPowerEventCallbacks.EvtDevicePrepareHardware = PrepareHardware;
    lPnpPowerEventCallbacks.EvtDeviceReleaseHardware = ReleaseHardware;

    WdfDeviceInitSetPnpPowerEventCallbacks(aDeviceInit, & lPnpPowerEventCallbacks);

}

// ===== Entry points =======================================================

// Thread  PnP
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

// Thread  PnP
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

// Threads  Users
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

// Threads  Users
void FileClose(WDFFILEOBJECT aFileObject)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aFileObject);

    (void)(aFileObject);
}

// Threads  Users
void FileCreate(WDFDEVICE aDevice, WDFREQUEST aRequest, WDFFILEOBJECT aFileObject)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice    );
    ASSERT(NULL != aRequest   );
    ASSERT(NULL != aFileObject);

    (void)(aDevice    );
    (void)(aFileObject);

    WdfRequestComplete(aRequest, STATUS_SUCCESS);
}

// Threads  Users
void IoInCallerContext(WDFDEVICE aDevice, WDFREQUEST aRequest)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice );
    ASSERT(NULL != aRequest);

    DeviceContext * lThis = GetDeviceContext(aDevice);

    lThis->mAdapter_WDF.IoInCallerContext(aRequest);
}

// Thread  PnP
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

// Thread  PnP
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
