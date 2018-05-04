
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Queue.c

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct _QUEUE_CONTEXT
{
    ULONG PrivateDeviceData;  // just a placeholder
}
QUEUE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(QUEUE_CONTEXT, QueueGetContext)

// Functions
/////////////////////////////////////////////////////////////////////////////

VOID Queue_IoDeviceControl(WDFQUEUE aQueue, WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aIoControlCode)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( , , %u bytes, %u bytes, 0x%08x )" DEBUG_EOL, aOutSize_byte, aInSize_byte, aIoControlCode);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    UNREFERENCED_PARAMETER(aQueue);

    WdfRequestComplete(aRequest, STATUS_SUCCESS);
}

VOID Queue_IoStop(WDFQUEUE aQueue, WDFREQUEST aRequest, ULONG aActionFlags)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, __FUNCTION__ "( , , 0x%08x )" DEBUG_EOL, aActionFlags);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    UNREFERENCED_PARAMETER(aQueue  );
    UNREFERENCED_PARAMETER(aRequest);

    return;
}

