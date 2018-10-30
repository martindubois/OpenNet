
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_NDIS/NdisDevice.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONDK_NDIS ==========================================================
#include "NdisAdapter.h"

#include "NdisDevice.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    void * mAdapter   ;
}
NdisDeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(NdisDeviceContext, GetNdisDeviceContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_DEVICE_D0_ENTRY D0Entry;
    static EVT_WDF_DEVICE_D0_EXIT  D0Exit ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS NdisDevice_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    NTSTATUS lResult = NetAdapterDeviceInitConfig(aDeviceInit);
    {
        WDF_OBJECT_ATTRIBUTES lAttributes;

        WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, NdisDeviceContext);

        WDF_PNPPOWER_EVENT_CALLBACKS lPnpPowerCallbacks;

        WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&lPnpPowerCallbacks);

        lPnpPowerCallbacks.EvtDeviceD0Entry = D0Entry;
        lPnpPowerCallbacks.EvtDeviceD0Exit  = D0Exit ;

        WdfDeviceInitSetPnpPowerEventCallbacks(aDeviceInit, &lPnpPowerCallbacks);

        WDFDEVICE lDevice;

        lResult = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
        if (STATUS_SUCCESS == lResult)
        {
            ASSERT(NULL != lDevice);

            NdisDeviceContext * lThis = GetNdisDeviceContext(lDevice);
            ASSERT(NULL != lThis);

            lResult = NdisAdapter_Create(lDevice, &lThis->mAdapter);
        }
        else
        {
            DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfDeviceCreate( , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
        }
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

    // TODO  ONK_NDIS
    //       Normal (Feature)

    return STATUS_SUCCESS;
}

NTSTATUS D0Exit(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aTargetState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice     );
    ASSERT(NULL != aTargetState);

    UNREFERENCED_PARAMETER(aDevice     );
    UNREFERENCED_PARAMETER(aTargetState);

    // TODO  ONK_NDIS
    //       Normal (Feature)

    return STATUS_SUCCESS;
}
