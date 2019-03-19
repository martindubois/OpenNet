
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Queue.cpp

// TEST COVERAGE  2019-03-29  KMS - Martin Dubois, ing.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_WDF.h>
#include <OpenNetK/Tunnel.h>

// ===== ONK_Tunnel_IO ======================================================
#include "VirtualHardware.h"

#include "Queue.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct _QUEUE_CONTEXT
{
    OpenNetK::Adapter_WDF * mAdapter ;
    VirtualHardware       * mHardware;
}
QueueContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(QueueContext, GetQueueContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL IoDeviceControl;
    EVT_WDF_IO_QUEUE_IO_READ           IoRead         ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

// aDevice   [---;RW-]
// aAdapter  [-K-;RW-]
// aHardware [-K-;RW-]
//
// Return  STATUS_SUCCESS
//         See WdfIoQueueCreate
//
// Thread  PnP

// NOT TESTED  Driver.LoadUnload.ErrorHandling
//             WdfIoQueueCreate fail

#pragma alloc_text (PAGE, Queue_Create)

NTSTATUS Queue_Create(WDFDEVICE aDevice, OpenNetK::Adapter_WDF * aAdapter, VirtualHardware * aHardware)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice  );
    ASSERT(NULL != aAdapter );
    ASSERT(NULL != aHardware);

    PAGED_CODE();

    WDF_IO_QUEUE_CONFIG lConfig;

    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&lConfig, WdfIoQueueDispatchSequential);

    lConfig.EvtIoDeviceControl = IoDeviceControl;
    lConfig.EvtIoRead          = IoRead         ;

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, QueueContext);

    WDFQUEUE lQueue;

    NTSTATUS lResult = WdfIoQueueCreate(aDevice, &lConfig, &lAttributes, &lQueue);
    if (STATUS_SUCCESS == lResult)
    {
        ASSERT(NULL != lQueue);

        QueueContext * lThis = GetQueueContext(lQueue);
        ASSERT(NULL != lThis);

        lThis->mAdapter  = aAdapter ;
        lThis->mHardware = aHardware;
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

VOID IoDeviceControl(WDFQUEUE aQueue, WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aIoControlCode)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , , %u bytes, %u bytes, 0x%08x )" DEBUG_EOL, aOutSize_byte, aInSize_byte, aIoControlCode);

    ASSERT(NULL != aQueue  );
    ASSERT(NULL != aRequest);

    QueueContext * lThis = GetQueueContext(aQueue);
    ASSERT(NULL != lThis          );
    ASSERT(NULL != lThis->mAdapter);

    lThis->mAdapter->IoDeviceControl(aRequest, aOutSize_byte, aInSize_byte, aIoControlCode);
}

void IoRead(WDFQUEUE aQueue, WDFREQUEST aRequest, size_t aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , , %u bytes )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL != aQueue    );
    ASSERT(NULL != aRequest  );
    ASSERT(   0 <  aSize_byte);

    QueueContext * lThis = GetQueueContext(aQueue);
    ASSERT(NULL != lThis           );
    ASSERT(NULL != lThis->mHardware);

    PVOID  lOut;
    size_t lOutSize_byte;

    NTSTATUS lStatus = WdfRequestRetrieveOutputBuffer(aRequest, sizeof(OpenNetK::Tunnel_PacketHeader) + 1, &lOut, &lOutSize_byte);
    if (STATUS_SUCCESS == lStatus)
    {
        unsigned int lInfo_byte = lThis->mHardware->Read(lOut, static_cast<unsigned int>(lOutSize_byte));

        WdfRequestCompleteWithInformation(aRequest, lStatus, lInfo_byte);
    }
    else
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ " - WdfRequestRetrieveOutputBuffer( , , ,  ) failed - 0x%08x" DEBUG_EOL, lStatus);

        WdfRequestComplete(aRequest, lStatus);
    }
}
