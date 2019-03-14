
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Device.cpp

// TODO ONK_Intel
//      Normal (Cleanup) - Move all what can be moved from Device.cpp to
//      OpenNetK::Hardware_WDF.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_WDF.h>
#include <OpenNetK/Hardware_WDF.h>
#include <OpenNetK/Interface.h>

// ===== ONK_Pro1000 ========================================================
#include "Queue.h"
#include "Intel_82576.h"
#include "Intel_82599.h"

#include "Device.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter      mAdapter     ;
    Intel                * mHardware    ;
    OpenNetK::Adapter_WDF  mAdapter_WDF ;
    OpenNetK::Hardware_WDF mHardware_WDF;

    union
    {
        Intel_82576              mIntel_82576;
        Intel_82599::Intel_82599 mIntel_82599;
    };
}
DeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DeviceContext, GetDeviceContext)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static NTSTATUS Init(DeviceContext * aThis, WDFDEVICE aDevice);

static void DeviceInit_Init(PWDFDEVICE_INIT aDeviceInit);

static unsigned short RetrieveDeviceId(WDFDEVICE aDevice);

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

// NOT TESTED  ONK_Intel.Device.ErrorHandling
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

// NOT TESTED  ONK_Intel.Device.ErrorHandling
//             Hardware_WDF.Init fail
NTSTATUS Init(DeviceContext * aThis, WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aThis);
    ASSERT(NULL != aDevice);

    // TODO  OpenNetK.Adapter
    //       Normal (Feature) - Add the PCI device and vendor ID to the
    //       information about the adapter.
    unsigned short lDeviceId = RetrieveDeviceId(aDevice);

    switch (lDeviceId)
    {
    case 0x10c9: aThis->mHardware = new (&aThis->mIntel_82576)              Intel_82576(); break;
    case 0x10fb: aThis->mHardware = new (&aThis->mIntel_82599) Intel_82599::Intel_82599(); break;

    default: ASSERT(false);
    }

    ASSERT(NULL != aThis->mHardware);

    NTSTATUS lResult = aThis->mHardware_WDF.Init(aDevice, aThis->mHardware);
    if (STATUS_SUCCESS == lResult)
    {
        aThis->mAdapter_WDF.Init(&aThis->mAdapter, aDevice, &aThis->mHardware_WDF);
        aThis->mAdapter.SetHardware(aThis->mHardware);

        aThis->mHardware->SetAdapter(&aThis->mAdapter);
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - Hardware_WDF::Init( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
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

// aDevice [---;RW-]
//
// Return  This method returns the PCI device id of the controller card.
unsigned short RetrieveDeviceId(WDFDEVICE aDevice)
{
    ASSERT(NULL != aDevice);

    WDFIOTARGET lTarget = WdfDeviceGetIoTarget(aDevice);
    ASSERT(NULL != lTarget);

    WDFREQUEST lRequest;

    NTSTATUS lStatus = WdfRequestCreate(WDF_NO_OBJECT_ATTRIBUTES, lTarget, &lRequest);
    ASSERT(STATUS_SUCCESS == lStatus);
    (void)(lStatus);

    unsigned short lBuffer[2];

    IO_STACK_LOCATION lStack;

    memset(&lStack, 0, sizeof(lStack));

    lStack.MajorFunction = IRP_MJ_PNP        ;
    lStack.MinorFunction = IRP_MN_READ_CONFIG;

    lStack.Parameters.ReadWriteConfig.Buffer     = lBuffer;
    lStack.Parameters.ReadWriteConfig.Length     = 4;
    lStack.Parameters.ReadWriteConfig.Offset     = 0;
    lStack.Parameters.ReadWriteConfig.WhichSpace = PCI_WHICHSPACE_CONFIG;

    WdfRequestWdmFormatUsingStackLocation(lRequest, &lStack);

    WDF_REQUEST_SEND_OPTIONS lOptions;

    WDF_REQUEST_SEND_OPTIONS_INIT(&lOptions, WDF_REQUEST_SEND_OPTION_SYNCHRONOUS | WDF_REQUEST_SEND_OPTION_IGNORE_TARGET_STATE);

    BOOLEAN lRetB = WdfRequestSend(lRequest, lTarget, &lOptions);
    ASSERT(lRetB);
    (void)(lRetB);

    WdfObjectDelete(lRequest);

    ASSERT(0x8086 == lBuffer[0]);

    return lBuffer[1];
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
    // DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

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
    // DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aFileObject);

    (void)(aFileObject);
}

// Threads  Users
void FileCreate(WDFDEVICE aDevice, WDFREQUEST aRequest, WDFFILEOBJECT aFileObject)
{
    // DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

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
    // DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

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
