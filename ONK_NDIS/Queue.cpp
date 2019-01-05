
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Queue.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_WDF.h>

// ===== ONK_NDIS ===========================================================
#include "Queue.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct _QUEUE_CONTEXT
{
    OpenNetK::Adapter_WDF * mAdapter;
}
QueueContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(QueueContext, GetQueueContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL IoDeviceControl;
    EVT_WDF_IO_QUEUE_IO_STOP           IoStop         ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// Thread  PnP

#pragma alloc_text (PAGE, Queue_Create)

// aDevice  [---;RW-] The WDFDEVICE instance
// aAdapter [-K-;RW-] The OpenNetK::Adapter_WDF instance
void Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice );
    ASSERT(NULL != aAdapter);

    PAGED_CODE();

    WDF_IO_QUEUE_CONFIG lConfig;

    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&lConfig, WdfIoQueueDispatchSequential);

    lConfig.EvtIoDeviceControl = IoDeviceControl;
    lConfig.EvtIoStop          = IoStop         ;

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, QueueContext);

    WDFQUEUE lQueue;

    NTSTATUS lStatus = WdfIoQueueCreate(aDevice, &lConfig, &lAttributes, &lQueue);
    ASSERT(STATUS_SUCCESS == lStatus);
    ASSERT(NULL           != lQueue );
    (void)(lStatus);

    QueueContext * lThis = GetQueueContext(lQueue);
    ASSERT(NULL != lThis);

    lThis->mAdapter = aAdapter;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

// Thread  Queue

VOID IoDeviceControl(WDFQUEUE aQueue, WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aIoControlCode)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( , , %u bytes, %u bytes, 0x%08x )" DEBUG_EOL, aOutSize_byte, aInSize_byte, aIoControlCode);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    QueueContext * lThis = GetQueueContext(aQueue);
    ASSERT(NULL != lThis          );
    ASSERT(NULL != lThis->mAdapter);

    lThis->mAdapter->IoDeviceControl(aRequest, aOutSize_byte, aInSize_byte, aIoControlCode);
}

VOID IoStop(WDFQUEUE aQueue, WDFREQUEST aRequest, ULONG aActionFlags)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( , , 0x%08x )" DEBUG_EOL, aActionFlags);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    UNREFERENCED_PARAMETER(aQueue  );
    UNREFERENCED_PARAMETER(aRequest);
}
