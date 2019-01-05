
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/NdisAdapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONK_NDIS ===========================================================
#include "Rx.h"
#include "Tx.h"

#include "NdisAdapter.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    VirtualHardware * mHardware;
}
AdapterContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(AdapterContext, GetAdapterContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================
extern "C"
{
    static EVT_NET_ADAPTER_CREATE_RXQUEUE       CreateRxQueue            ;
    static EVT_NET_ADAPTER_CREATE_TXQUEUE       CreateTxQueue            ;
    static EVT_NET_ADAPTER_SET_CAPABILITIES     SetCapabilities          ;
    static EVT_TXQUEUE_ADVANCE                  Tx_Advance               ;
    static EVT_TXQUEUE_CANCEL                   Tx_Cancel                ;
    static EVT_TXQUEUE_SET_NOTIFICATION_ENABLED Tx_SetNotificationEnabled;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// aDevice   [---;R--]
// aAdapter  [---;-W-] The function return the adapter context here.
// aHardware [-K-;RW-] The VirtualHardware instance
NTSTATUS NdisAdapter_Create(WDFDEVICE aDevice, void ** aAdapter, VirtualHardware * aHardware)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice  );
    ASSERT(NULL != aAdapter );
    ASSERT(NULL != aHardware);

    NET_ADAPTER_CONFIG lConfig;

    NET_ADAPTER_CONFIG_INIT(&lConfig, SetCapabilities, CreateTxQueue, CreateRxQueue);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, AdapterContext);

    NETADAPTER lAdapter;

    DbgBreakPoint();

    NTSTATUS lResult = NetAdapterCreate(aDevice, &lAttributes, &lConfig, &lAdapter);
    if (STATUS_SUCCESS == lResult)
    {
        ASSERT(NULL != lAdapter);

        AdapterContext * lThis = GetAdapterContext(lAdapter);
        ASSERT(NULL != lThis);

        lThis->mHardware = aHardware;

        (*aAdapter) = lThis;
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS CreateRxQueue(NETADAPTER aAdapter, PNETRXQUEUE_INIT aQueueInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aAdapter  );
    ASSERT(NULL != aQueueInit);

    AdapterContext * lThis = GetAdapterContext(aAdapter);
    ASSERT(NULL != lThis);

    Rx_Create(aQueueInit, lThis->mHardware);

    return STATUS_SUCCESS;
}

NTSTATUS CreateTxQueue(NETADAPTER aAdapter, PNETTXQUEUE_INIT aQueueInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aAdapter  );
    ASSERT(NULL != aQueueInit);

    AdapterContext * lThis = GetAdapterContext(aAdapter);
    ASSERT(NULL != lThis);

    Tx_Create(aQueueInit, lThis->mHardware);

    return STATUS_SUCCESS;
}

NTSTATUS SetCapabilities(NETADAPTER aAdapter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aAdapter);

    NET_ADAPTER_POWER_CAPABILITIES lPowerCapabilities;

    NET_ADAPTER_POWER_CAPABILITIES_INIT(&lPowerCapabilities);

    lPowerCapabilities.SupportedWakePatterns = NET_ADAPTER_WAKE_MAGIC_PACKET;

    NetAdapterSetPowerCapabilities(aAdapter, &lPowerCapabilities);

    // TODO  ONK_NDIS
    //       Normal (Feature)

    return STATUS_SUCCESS;
}
