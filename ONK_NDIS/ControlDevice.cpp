
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_NDIS/ControlDevice.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_WDF.h>
#include <OpenNetK/Interface.h>

// ===== ONDK_NDIS =======================================================
#include "Queue.h"

#include "ControlDevice.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter     mAdapter    ;
    OpenNetK::Adapter_WDF mAdapter_WDF;
}
ControlDeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(ControlDeviceContext, GetControlDeviceContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static void DeviceInit_Init(WDFDEVICE_INIT * aDeviceInit);

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_FILE_CLEANUP         FileCleanup      ;
    static EVT_WDF_FILE_CLOSE           FileClose        ;
    static EVT_WDF_DEVICE_FILE_CREATE   FileCreate       ;
    static EVT_WDF_IO_IN_CALLER_CONTEXT IoInCallerContext;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

#pragma alloc_text (PAGE, ControlDevice_Create)

// aDeviceInit [---;RW-]
void ControlDevice_Create(WDFDEVICE_INIT * aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    DeviceInit_Init(aDeviceInit);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, ControlDeviceContext);

    WDFDEVICE lDevice;

    NTSTATUS lStatus = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
    ASSERT(STATUS_SUCCESS == lStatus);
    ASSERT(NULL           != lDevice);

    ControlDeviceContext * lThis = GetControlDeviceContext(lDevice);
    ASSERT(NULL != lThis);

    lStatus = WdfDeviceCreateDeviceInterface(lDevice, &OPEN_NET_DRIVER_INTERFACE, NULL);
    ASSERT(STATUS_SUCCESS == lStatus);

    Queue_Create(lDevice, &lThis->mAdapter_WDF);
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aDeviceInit [---;RW-]
void DeviceInit_Init(WDFDEVICE_INIT * aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    WDF_FILEOBJECT_CONFIG lFileObjectConfig;

    WDF_FILEOBJECT_CONFIG_INIT(&lFileObjectConfig, FileCreate, FileClose, FileCleanup);

    WdfDeviceInitSetFileObjectConfig(aDeviceInit, &lFileObjectConfig, WDF_NO_OBJECT_ATTRIBUTES);

    WdfDeviceInitSetIoInCallerContextCallback(aDeviceInit, IoInCallerContext);
}

// ===== Entry points =======================================================

// Threads  Users
void FileCleanup(WDFFILEOBJECT aFileObject)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aFileObject);

    WDFDEVICE lDevice = WdfFileObjectGetDevice(aFileObject);
    ASSERT(NULL != lDevice);

    ControlDeviceContext * lThis = GetControlDeviceContext(lDevice);
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
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

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

    ASSERT(NULL != aDevice);
    ASSERT(NULL != aRequest);

    ControlDeviceContext * lThis = GetControlDeviceContext(aDevice);

    lThis->mAdapter_WDF.IoInCallerContext(aRequest);
}
