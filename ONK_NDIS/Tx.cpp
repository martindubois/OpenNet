
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Tx.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONK_NDIS ===========================================================
#include "Tx.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    VirtualHardware * mHardware;
}
TxContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(TxContext, GetTxContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================
extern "C"
{
    static EVT_TXQUEUE_ADVANCE                  Advance               ;
    static EVT_TXQUEUE_CANCEL                   Cancel                ;
    static EVT_TXQUEUE_SET_NOTIFICATION_ENABLED SetNotificationEnabled;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// aQueueInit [---;RW-]
// aHardware  [-K-;RW-]
void Tx_Create(NETTXQUEUE_INIT * aQueueInit, VirtualHardware * aHardware)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aQueueInit);
    ASSERT(NULL != aHardware );

    NET_TXQUEUE_CONFIG lConfig;

    NET_TXQUEUE_CONFIG_INIT(&lConfig, Advance, SetNotificationEnabled, Cancel);

    NETTXQUEUE lQueue;

    NTSTATUS lStatus = NetTxQueueCreate(aQueueInit, WDF_NO_OBJECT_ATTRIBUTES, &lConfig, &lQueue);
    ASSERT(STATUS_SUCCESS == lStatus);
    ASSERT(NULL           != lQueue );
    (void)(lStatus);

    TxContext * lThis = GetTxContext(lQueue);
    ASSERT(NULL != lThis);

    lThis->mHardware = aHardware;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

VOID Advance(NETTXQUEUE aQueue)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aQueue);

    TxContext * lThis = GetTxContext(aQueue);
    ASSERT(NULL != lThis);

    const NET_DATAPATH_DESCRIPTOR * lDesc = NetTxQueueGetDatapathDescriptor(aQueue);
    ASSERT(NULL != lDesc);

    NET_PACKET * lPacket;

    while (NULL != (lPacket = NetRingBufferGetNextPacket(lDesc)))
    {
        NET_PACKET_FRAGMENT * lFragment = NET_PACKET_GET_FRAGMENT(lPacket, lDesc, 0);
        ASSERT(NULL != lFragment                );
        ASSERT(     !  lFragment->Completed     );
        ASSERT(   0 <  lFragment->ValidLength   );
        ASSERT(NULL != lFragment->VirtualAddress);

        lThis->mHardware->Rx_IndicatePacket(lFragment->VirtualAddress, lFragment->ValidLength);

        lFragment->Completed = true;

        NetRingBufferAdvanceNextPacket(lDesc);
    }

    NetRingBufferReturnCompletedPackets(lDesc);
}

VOID Cancel(NETTXQUEUE aQueue)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aQueue);

    UNREFERENCED_PARAMETER(aQueue);
}

VOID SetNotificationEnabled(NETTXQUEUE aQueue, BOOLEAN aEnabled)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , %s )" DEBUG_EOL, aEnabled ? "true" : "false");

    ASSERT(NULL != aQueue);

    UNREFERENCED_PARAMETER(aQueue);
}
