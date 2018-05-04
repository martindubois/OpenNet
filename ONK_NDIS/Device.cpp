
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_NDIS/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== ONDK_NDIS ==========================================================
#include "Adapter.h"

#include "Device.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    void * mAdapter;
}
DEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, GetDeviceContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_DEVICE_D0_ENTRY         D0Entry        ;
    static EVT_WDF_DEVICE_D0_EXIT          D0Exit         ;
    static EVT_WDF_DEVICE_PREPARE_HARDWARE PrepareHardware;
    static EVT_WDF_DEVICE_RELEASE_HARDWARE ReleaseHardware;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS Device_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    NTSTATUS lResult; // = NetAdapterDeviceInitConfig(aDeviceInit);
//    if (STATUS_SUCCESS == lResult)
    {
        WDF_OBJECT_ATTRIBUTES lAttributes;
        WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, DEVICE_CONTEXT);

        WDF_PNPPOWER_EVENT_CALLBACKS lPnpPowerCallbacks;

        WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&lPnpPowerCallbacks);

        lPnpPowerCallbacks.EvtDeviceD0Entry         = D0Entry        ;
        lPnpPowerCallbacks.EvtDeviceD0Exit          = D0Exit         ;
        lPnpPowerCallbacks.EvtDevicePrepareHardware = PrepareHardware;
        lPnpPowerCallbacks.EvtDeviceReleaseHardware = ReleaseHardware;

        WdfDeviceInitSetPnpPowerEventCallbacks(aDeviceInit, &lPnpPowerCallbacks);

        WDFDEVICE lDevice;

        lResult = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
        if (STATUS_SUCCESS == lResult)
        {
            ASSERT(NULL != lDevice);

            void * lAdapter;

            lResult = Adapter_Create(lDevice, &lAdapter);

            DEVICE_CONTEXT * lThis = GetDeviceContext(lDevice);
            ASSERT(NULL != lThis);

            lThis->mAdapter = lAdapter;
        }
/*        else
        {
            DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDeviceCreate( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
        } */
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS D0Entry(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aPreviousState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice       );
    ASSERT(NULL != aPreviousState);

    UNREFERENCED_PARAMETER(aDevice       );
    UNREFERENCED_PARAMETER(aPreviousState);

    // TODO Dev

    return STATUS_SUCCESS;
}

NTSTATUS D0Exit(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aTargetState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice     );
    ASSERT(NULL != aTargetState);

    UNREFERENCED_PARAMETER(aDevice     );
    UNREFERENCED_PARAMETER(aTargetState);

    // TODO Dev

    return STATUS_SUCCESS;
}

NTSTATUS PrepareHardware(WDFDEVICE aDevice, WDFCMRESLIST aResourcesRaw, WDFCMRESLIST aResourcesTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice             );
    ASSERT(NULL != aResourcesRaw       );
    ASSERT(NULL != aResourcesTranslated);

    UNREFERENCED_PARAMETER(aDevice             );
    UNREFERENCED_PARAMETER(aResourcesRaw       );
    UNREFERENCED_PARAMETER(aResourcesTranslated);

    // TODO Dev

    return STATUS_SUCCESS;
}

NTSTATUS ReleaseHardware(WDFDEVICE aDevice, WDFCMRESLIST aResourcesTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice             );
    ASSERT(NULL != aResourcesTranslated);

    UNREFERENCED_PARAMETER(aDevice             );
    UNREFERENCED_PARAMETER(aResourcesTranslated);

    // TODO Dev

    return STATUS_SUCCESS;
}