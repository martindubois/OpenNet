
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/NdisAdapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONK_NDIS ===========================================================
#include "NdisAdapter.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    ULONG PrivateDeviceData;  // just a placeholder
}
ADAPTER_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(ADAPTER_CONTEXT, GetAdapterContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static NTSTATUS InitContext     (ADAPTER_CONTEXT * aThis);
static NTSTATUS InitRequestQueue(ADAPTER_CONTEXT * aThis);

// ===== Entry points =======================================================
extern "C"
{
    static EVT_NET_ADAPTER_CREATE_RXQUEUE   CreateRxQueue  ;
    static EVT_NET_ADAPTER_CREATE_TXQUEUE   CreateTxQueue  ;
    static EVT_NET_ADAPTER_SET_CAPABILITIES SetCapabilities;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS NdisAdapter_Create(WDFDEVICE aDevice, void ** aAdapter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDevice );
    ASSERT(NULL != aAdapter);

    UNREFERENCED_PARAMETER(aDevice );
    UNREFERENCED_PARAMETER(aAdapter);

    NET_ADAPTER_CONFIG lConfig;

    NET_ADAPTER_CONFIG_INIT(&lConfig, SetCapabilities, CreateTxQueue, CreateRxQueue);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, ADAPTER_CONTEXT);

    NETADAPTER lAdapter;

    NTSTATUS lResult = NetAdapterCreate(aDevice, &lAttributes, &lConfig, &lAdapter);
    if (STATUS_SUCCESS == lResult)
    {
        ASSERT(NULL != lAdapter);

        ADAPTER_CONTEXT * lThis = GetAdapterContext(lAdapter);
        ASSERT(NULL != lThis);

        lResult = InitContext(lThis);
        if (STATUS_SUCCESS == lResult)
        {
            lResult = InitRequestQueue(lThis);
            if (STATUS_SUCCESS == lResult)
            {
                (*aAdapter) = lThis;
            }
        }
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - NetAdapterCreate( , , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS InitContext(ADAPTER_CONTEXT * aThis)
{
    ASSERT(NULL != aThis);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aThis);

    return STATUS_SUCCESS;
}

NTSTATUS InitRequestQueue(ADAPTER_CONTEXT * aThis)
{
    ASSERT(NULL != aThis);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aThis);

    return STATUS_SUCCESS;
}

// ===== Entry points =======================================================

NTSTATUS CreateRxQueue(NETADAPTER aAdapter, PNETRXQUEUE_INIT aRxQueueInit)
{
    ASSERT(NULL != aAdapter    );
    ASSERT(NULL != aRxQueueInit);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aAdapter    );
    (void)(aRxQueueInit);

    return STATUS_SUCCESS;
}

NTSTATUS CreateTxQueue(NETADAPTER aAdapter,	PNETTXQUEUE_INIT aTxQueueInit)
{
    ASSERT(NULL != aAdapter    );
    ASSERT(NULL != aTxQueueInit);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aAdapter    );
    (void)(aTxQueueInit);

    return STATUS_SUCCESS;
}

NTSTATUS SetCapabilities(NETADAPTER aAdapter)
{
    ASSERT(NULL != aAdapter);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aAdapter);

    return STATUS_SUCCESS;
}