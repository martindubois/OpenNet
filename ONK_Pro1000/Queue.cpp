
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Queue.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_WDF.h>

// ===== ONK_Pro1000 ========================================================
#include "Queue.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct _QUEUE_CONTEXT
{
    OpenNetK::Adapter_WDF * mAdapter;
}
QueueContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(QueueContext, QueueGetContext)

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

// NOT TESTED  ONK_Hardware.Queue.ErrorHandling
//             WdfIoQueueCreate fail

#pragma alloc_text (PAGE, Queue_Create)

NTSTATUS Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter)
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

    NTSTATUS lResult = WdfIoQueueCreate(aDevice, &lConfig, &lAttributes, &lQueue);
    if (STATUS_SUCCESS == lResult)
    {
        ASSERT(NULL != lQueue);

        QueueContext * lThis = QueueGetContext(lQueue);
        ASSERT(NULL != lThis);

        lThis->mAdapter = aAdapter;
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ERROR, PREFIX __FUNCTION__ " - WdfIoQueueCreate( , , ,  ) failed - 0x%08x" DEBUG_EOL, lResult);
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

// Thread  Queue

VOID IoDeviceControl(WDFQUEUE aQueue, WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aIoControlCode)
{
    // DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( , , %u bytes, %u bytes, 0x%08x )" DEBUG_EOL, aOutSize_byte, aInSize_byte, aIoControlCode);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    QueueContext * lThis = QueueGetContext(aQueue);
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
