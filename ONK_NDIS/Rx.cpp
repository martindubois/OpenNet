
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Rx.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONK_NDIS ===========================================================
#include "Rx.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    VirtualHardware * mHardware;
    NETRXQUEUE        mQueue   ;

    // ===== Zone 0 =========================================================
    WDFSPINLOCK  mZone0;

    unsigned int mIndex              ;
    bool         mNotificationEnabled;
}
RxContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(RxContext, GetRxContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static bool Indicate(RxContext * aThis,                                           const void * aData, unsigned int aSize_byte);
static void Indicate(const NET_DATAPATH_DESCRIPTOR * aDesc, NET_PACKET * aPacket, const void * aData, unsigned int aSize_byte);

// ===== Entry points =======================================================

static VirtualHardware::Tx_Callback Indicate;

extern "C"
{
    static EVT_RXQUEUE_ADVANCE                  Advance               ;
    static EVT_RXQUEUE_CANCEL                   Cancel                ;
    static EVT_RXQUEUE_SET_NOTIFICATION_ENABLED SetNotificationEnabled;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// aQueueInit [---;RW-]
// aHardware  [-K-;RW-]
void Rx_Create(NETRXQUEUE_INIT * aQueueInit, VirtualHardware * aHardware)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aQueueInit);
    ASSERT(NULL != aHardware );

    NET_RXQUEUE_CONFIG lConfig;

    NET_RXQUEUE_CONFIG_INIT(&lConfig, Advance, SetNotificationEnabled, Cancel);

    NETRXQUEUE lQueue;

    NTSTATUS lStatus = NetRxQueueCreate(aQueueInit, WDF_NO_OBJECT_ATTRIBUTES, &lConfig, &lQueue);
    ASSERT(STATUS_SUCCESS == lStatus);
    ASSERT(NULL           != lQueue );

    RxContext * lThis = GetRxContext(lQueue);
    ASSERT(NULL != lThis);

    lThis->mHardware            = aHardware ;
    lThis->mIndex               = 0xffffffff;
    lThis->mQueue               = lQueue    ;
    lThis->mNotificationEnabled = false     ;

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT(&lAttributes);

    lAttributes.ParentObject = lQueue;

    lStatus = WdfSpinLockCreate(&lAttributes, &lThis->mZone0);
    ASSERT(STATUS_SUCCESS == lStatus      );
    ASSERT(NULL           != lThis->mZone0);

    lThis->mHardware->Tx_RegisterCallback(Indicate, lThis);
}

// ===== Entry point ========================================================

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aThis   [---;RW-]
// aData   [---;R--]
// aSize_byte
//
// Return
//  false  The packet has been dropped
//  true   The packet is transmitted
bool Indicate(RxContext * aThis, const void * aData, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , , %u bytes )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL       != aThis        );
    ASSERT(0xffffffff != aThis->mIndex);
    ASSERT(NULL       != aThis->mQueue);
    ASSERT(NULL       != aData        );
    ASSERT(         0 <  aSize_byte   );

    bool lResult;

    const NET_DATAPATH_DESCRIPTOR * lDesc = NetRxQueueGetDatapathDescriptor(aThis->mQueue);
    ASSERT(NULL != lDesc);

    NET_RING_BUFFER * lRing = NET_DATAPATH_DESCRIPTOR_GET_PACKET_RING_BUFFER(lDesc);
    ASSERT(NULL != lRing);

    if (lRing->NextIndex == aThis->mIndex)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_WARNING, PREFIX __FUNCTION__ " - Dropped packet" DEBUG_EOL);
        lResult = false;
    }
    else
    {
        NET_PACKET * lPacket = NetRingBufferGetPacketAtIndex(lDesc, aThis->mIndex);
        ASSERT(NULL != lPacket);

        Indicate(lDesc, lPacket, aData, aSize_byte);

        aThis->mIndex = NetRingBufferIncrementIndex(lRing, aThis->mIndex);

        lResult = true;
    }

    return lResult;
}

// aDesc   [---;R--]
// aPacket [---;RW-]
// aData   [---;R--]
// aSize_byte
void Indicate(const NET_DATAPATH_DESCRIPTOR * aDesc, NET_PACKET * aPacket, const void * aData, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , , , %u bytes )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL != aDesc     );
    ASSERT(NULL != aPacket   );
    ASSERT(NULL != aData     );
    ASSERT(   0 <  aSize_byte);

    NET_PACKET_FRAGMENT * lFragment = NET_PACKET_GET_FRAGMENT(aPacket, aDesc, 0);
    ASSERT(NULL != lFragment                );
    ASSERT(NULL != lFragment->VirtualAddress);

    memcpy(lFragment->VirtualAddress, aData, aSize_byte);

    lFragment->ValidLength         = aSize_byte;
    lFragment->Offset              =          0;
    lFragment->LastFragmentOfFrame = true      ;
}

// ===== Entry points =======================================================

bool Indicate(void * aContext, const void * aData, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , , %u bytes )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL != aContext  );
    ASSERT(NULL != aData     );
    ASSERT(0    <  aSize_byte);

    RxContext * lThis = reinterpret_cast<RxContext *>(aContext);
    ASSERT(NULL != lThis        );
    ASSERT(NULL != lThis->mQueue);
    ASSERT(NULL != lThis->mZone0);

    bool lResult;

    WdfSpinLockAcquire(lThis->mZone0);

        if (0xffffffff == lThis->mIndex)
        {
            DbgPrintEx(DEBUG_ID, DEBUG_WARNING, PREFIX __FUNCTION__ " - Dropped packet (Not initialized)" DEBUG_EOL);
            lResult = false;
        }
        else
        {
            lResult = Indicate(lThis, aData, aSize_byte);
        }

        if (lThis->mNotificationEnabled)
        {
            NetRxQueueNotifyMoreReceivedPacketsAvailable(lThis->mQueue);
        }

    WdfSpinLockRelease(lThis->mZone0);

    return lResult;
}

VOID Advance(NETRXQUEUE aQueue)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aQueue);

    RxContext * lThis = GetRxContext(aQueue);
    ASSERT(NULL != lThis);

    const NET_DATAPATH_DESCRIPTOR * lDesc = NetRxQueueGetDatapathDescriptor(aQueue);
    ASSERT(NULL != lDesc);

    NET_RING_BUFFER * lRing = NET_DATAPATH_DESCRIPTOR_GET_PACKET_RING_BUFFER(lDesc);
    ASSERT(NULL != lRing);

    WdfSpinLockAcquire(lThis->mZone0);

        if (0xffffffff == lThis->mIndex)
        {
            lThis->mIndex = lRing->BeginIndex;
        }
        else
        {
            lRing->BeginIndex = lThis->mIndex;
        }

    WdfSpinLockRelease(lThis->mZone0);
}

VOID Cancel(NETRXQUEUE aQueue)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aQueue);

    const NET_DATAPATH_DESCRIPTOR * lDesc = NetRxQueueGetDatapathDescriptor(aQueue);
    ASSERT(NULL != lDesc);

    NetRingBufferReturnAllPackets(lDesc);
}

VOID SetNotificationEnabled(NETRXQUEUE aQueue, BOOLEAN aEnabled)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , %s )" DEBUG_EOL, aEnabled ? "true" : "false");

    ASSERT(NULL != aQueue);

    RxContext * lThis = GetRxContext(aQueue);
    ASSERT(NULL != lThis        );
    ASSERT(NULL != lThis->mZone0);

    WdfSpinLockAcquire(lThis->mZone0);

        lThis->mNotificationEnabled = aEnabled;

    WdfSpinLockRelease(lThis->mZone0);
}
