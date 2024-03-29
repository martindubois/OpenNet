
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved
// Product    OpenNet
// File       ONK_Lib/Adapter_WDF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Hardware_WDF.h>

#include <OpenNetK/Adapter_WDF.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"

// ===== ONK_Lib ============================================================
#include "OSDep_WDF.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void ProcessEvent(void * aContext);

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter_WDF::Init(Adapter * aAdapter, WDFDEVICE aDevice, Hardware_WDF * aHardware_WDF)
    {
        ASSERT(NULL != aAdapter     );
        ASSERT(NULL != aDevice      );
        ASSERT(NULL != aHardware_WDF);

        OSDep_Init(&mOSDep, this);

        mAdapter      = aAdapter     ;
        mDevice       = aDevice      ;
        mHardware_WDF = aHardware_WDF;

        WDF_IO_QUEUE_CONFIG lConfig;

        WDF_IO_QUEUE_CONFIG_INIT(&lConfig, WdfIoQueueDispatchManual);

        NTSTATUS lStatus = WdfIoQueueCreate(aDevice, &lConfig, WDF_NO_OBJECT_ATTRIBUTES, &mWaiting);
        ASSERT(STATUS_SUCCESS == lStatus );
        ASSERT(NULL           != mWaiting);

        WDF_OBJECT_ATTRIBUTES lAttr;

        WDF_OBJECT_ATTRIBUTES_INIT(&lAttr);

        lAttr.ParentObject = aDevice;

        WDFSPINLOCK lSpinLock;

        lStatus = WdfSpinLockCreate(&lAttr, &lSpinLock);
        ASSERT(STATUS_SUCCESS == lStatus  );
        ASSERT(NULL           != lSpinLock);
        (void)(lStatus);

        mZone0.SetLock (lSpinLock);
        mZone0.SetOSDep(&mOSDep  );

        mAdapter->Init    (&mZone0);
        mAdapter->SetOSDep(&mOSDep);

        mAdapter->Event_RegisterCallback(ProcessEvent, this);
    }

    void Adapter_WDF::FileCleanup(WDFFILEOBJECT aFileObject)
    {
        ASSERT(NULL != aFileObject);

        ASSERT(NULL != mAdapter);

        mAdapter->FileCleanup( aFileObject );
    }

    // CRITICAL PATH  BufferEvent
    void Adapter_WDF::IoDeviceControl(WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aCode)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mAdapter);

        NTSTATUS lStatus = STATUS_NOT_SUPPORTED;

        OpenNetK_IoCtl_Info lInfo;

        if (mAdapter->IoCtl_GetInfo(aCode, &lInfo))
        {
            if ((aInSize_byte < lInfo.mIn_MinSize_byte) || (aOutSize_byte < lInfo.mOut_MinSize_byte))
            {
                lStatus = STATUS_INVALID_BUFFER_SIZE;
            }
            else
            {
                void * lIn = NULL;

                lStatus = (0 < aInSize_byte) ? WdfRequestRetrieveInputBuffer(aRequest, lInfo.mIn_MinSize_byte, &lIn, NULL) : STATUS_SUCCESS;
                if (STATUS_SUCCESS == lStatus)
                {
                    void * lOut = NULL;

                    lStatus = (0 < aOutSize_byte) ? WdfRequestRetrieveOutputBuffer(aRequest, lInfo.mOut_MinSize_byte, &lOut, NULL) : STATUS_SUCCESS;
                    if (STATUS_SUCCESS == lStatus)
                    {
                        int lRet = mAdapter->IoCtl(WdfRequestGetFileObject( aRequest ), aCode, lIn, static_cast<unsigned int>(aInSize_byte), lOut, static_cast<unsigned int>(aOutSize_byte));

                        ProcessIoCtlResult(lRet, aRequest);

                        lStatus = ResultToStatus(aRequest, lRet);
                    }
                }
            }
        }

        if (STATUS_PENDING != lStatus)
        {
            WdfRequestComplete(aRequest, lStatus);
        }
    }

    void Adapter_WDF::IoInCallerContext(WDFREQUEST aRequest)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mDevice);

        WDF_REQUEST_PARAMETERS lParameters;

        WDF_REQUEST_PARAMETERS_INIT(&lParameters);

        WdfRequestGetParameters(aRequest, &lParameters);

        if ( (WdfRequestTypeDeviceControl == lParameters.Type) && (IOCTL_CONNECT == lParameters.Parameters.DeviceIoControl.IoControlCode) )
        {
            NTSTATUS lStatus;

            IoCtl_Connect_In * lIn         ;
            size_t             lInSize_byte;

            lStatus = WdfRequestRetrieveInputBuffer(aRequest, sizeof(IoCtl_Connect_In), reinterpret_cast<PVOID *>(&lIn), &lInSize_byte);
            if (STATUS_SUCCESS == lStatus)
            {
                ASSERT(NULL                     != lIn         );
                ASSERT(sizeof(IoCtl_Connect_In) <= lInSize_byte);

                lStatus = Connect(lIn);
            }

            if (STATUS_SUCCESS != lStatus)
            {
                WdfRequestComplete(aRequest, lStatus);
                return;
            }
        }

        WdfDeviceEnqueueRequest(mDevice, aRequest);
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    // CRITICAL PATH  BufferEvent
    void Adapter_WDF::Event_Process()
    {
        ASSERT(NULL != mWaiting);

        WDFREQUEST lRequest;

        NTSTATUS lStatus = WdfIoQueueRetrieveNextRequest(mWaiting, &lRequest);
        if (STATUS_SUCCESS == lStatus)
        {
            ASSERT(NULL != lRequest);

            WDF_REQUEST_PARAMETERS lParameters;

            WDF_REQUEST_PARAMETERS_INIT(&lParameters);

            WdfRequestGetParameters(lRequest, &lParameters);

            IoDeviceControl(lRequest, lParameters.Parameters.DeviceIoControl.OutputBufferLength, lParameters.Parameters.DeviceIoControl.InputBufferLength, lParameters.Parameters.DeviceIoControl.IoControlCode);
        }
    }

    // SharedMemory_Translate ==> SharedMemory_Release
    //
    // Level   PASSIVE
    // Thread  Users
    void Adapter_WDF::SharedMemory_Release()
    {
        ASSERT(NULL != mSharedMemory_MDL);

        // MmProbeAndLockPages          ==> MmUnlockPages    See SharedMemory_Translate
        // MmGetSystemAddressForMdlSafe ==> MmUnlockedPages  See SharedMemory_Translate
        MmUnlockPages(mSharedMemory_MDL);

        // IoAllocateMdl ==> IoFreeMdl  See SharedMemory_Translate
        IoFreeMdl(mSharedMemory_MDL);

        mSharedMemory_MDL = NULL;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aIn [---;RW-] The input data
    //
    // Return  STATUS_SUCCESS
    //         See Event_Translate
    //         See SharedMemory_Translate
    //
    // Level    PASSIVE
    // Threads  Users
    NTSTATUS Adapter_WDF::Connect(void * aIn)
    {
        ASSERT(NULL != aIn        );

        IoCtl_Connect_In * lIn = reinterpret_cast<IoCtl_Connect_In *>(aIn);

        // SharedMemory_Translate ==> SharedMemory_Release  See FileCleanup
        return SharedMemory_Translate(&lIn->mSharedMemory);
    }

    // Level   DISPATCH
    // Thread  Queue
    //
    // CRITICAL PATH  BufferEvent  1+ / IOCTL_EVENT_WAIT
    void Adapter_WDF::ProcessIoCtlResult(int aIoCtlResult, WDFREQUEST aRequest)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mHardware_WDF);
        ASSERT(NULL != mWaiting     );

        OpenNetK_IoCtl_Result lIoCtlResult = static_cast<OpenNetK_IoCtl_Result>(aIoCtlResult);

        switch (lIoCtlResult)
        {
        case IOCTL_RESULT_INVALID_SYSTEM_ID:
        case IOCTL_RESULT_TOO_MANY_ADAPTER :
            // SharedMemory_Translate ==> SharedMemory_Release  See IoInCallerContext
            SharedMemory_Release();
            break;

        case IOCTL_RESULT_PROCESSING_NEEDED:
            mHardware_WDF->TrigProcess2();
            break;

        case IOCTL_RESULT_WAIT:
            WdfRequestForwardToIoQueue(aRequest, mWaiting);
            break;
        }
    }

    // Level   PASSIVE
    // Thread  Users
    NTSTATUS Adapter_WDF::SharedMemory_ProbeAndLock()
    {
        ASSERT(NULL != mSharedMemory_MDL);

        NTSTATUS lResult;

        __try
        {
            // MmProbeAndLockPages ==> MmUnlockPages  See SharedMemory_Release
            MmProbeAndLockPages(mSharedMemory_MDL, KernelMode, IoModifyAccess);
            lResult = STATUS_SUCCESS;
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            lResult = STATUS_INVALID_ADDRESS;
        }

        return lResult;
    }

    // aSharedMemory [DK-;RW-]
    //
    // SharedMemory_Translate ==> SharedMemory_Release
    //
    // Level    PASSIVE
    // Threads  Users
    NTSTATUS Adapter_WDF::SharedMemory_Translate(void ** aSharedMemory)
    {
        ASSERT(NULL != aSharedMemory);

        if (NULL == (*aSharedMemory)   )
        {
            return STATUS_INVALID_ADDRESS;
        }

        if (NULL != mSharedMemory_MDL)
        {
            return STATUS_INVALID_STATE_TRANSITION;
        }

        // IoAllocateMdl ==> IoFreeMdl  See SharedMemory_Release
        mSharedMemory_MDL = IoAllocateMdl(*aSharedMemory, SHARED_MEMORY_SIZE_byte, FALSE, FALSE, NULL);
        if (NULL == mSharedMemory_MDL)
        {
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        NTSTATUS lResult = SharedMemory_ProbeAndLock();
        if (STATUS_SUCCESS == lResult)
        {
            // MmGetSystemAddressForMdlSafe ==> MmUnlockedPages  See SharedMemory_Release
            (*aSharedMemory) = MmGetSystemAddressForMdlSafe(mSharedMemory_MDL, NormalPagePriority | MdlMappingNoExecute);
            if (NULL == (*aSharedMemory))
            {
                MmUnlockPages(mSharedMemory_MDL);
                lResult = STATUS_INSUFFICIENT_RESOURCES;
            }
        }

        if (STATUS_SUCCESS != lResult)
        {
            IoFreeMdl(mSharedMemory_MDL);
            mSharedMemory_MDL = NULL;
        }

        return lResult;
    }

    // CRITICAL PATH  1+ / IOCTL_EVENT_WAIT
    NTSTATUS Adapter_WDF::ResultToStatus(WDFREQUEST aRequest, int aIoCtlResult)
    {
        ASSERT(NULL != aRequest);

        OpenNetK_IoCtl_Result lIoCtlResult = static_cast<OpenNetK_IoCtl_Result>(aIoCtlResult);

        NTSTATUS lResult;

        switch (lIoCtlResult)
        {
        case IOCTL_RESULT_OK               :
        case IOCTL_RESULT_PROCESSING_NEEDED:
            lResult = STATUS_SUCCESS;
            break;

        case IOCTL_RESULT_ALREADY_CONNECTED: lResult = STATUS_ALREADY_COMMITTED       ; break;
        case IOCTL_RESULT_CANNOT_DROP      : lResult = STATUS_NO_MEMORY               ; break;
        case IOCTL_RESULT_CANNOT_MAP_BUFFER: lResult = STATUS_NO_MEMORY               ; break;
        case IOCTL_RESULT_CANNOT_SEND      : lResult = STATUS_NO_MEMORY               ; break;
        case IOCTL_RESULT_ERROR            : lResult = STATUS_UNSUCCESSFUL            ; break;
        case IOCTL_RESULT_INVALID_PARAMETER: lResult = STATUS_INVALID_PARAMETER       ; break;
        case IOCTL_RESULT_INVALID_SYSTEM_ID: lResult = STATUS_INVALID_SID             ; break;
        case IOCTL_RESULT_NO_BUFFER        : lResult = STATUS_NO_MEMORY               ; break;
        case IOCTL_RESULT_NOT_SET          : lResult = STATUS_NOT_COMMITTED           ; break;
        case IOCTL_RESULT_RUNNING          : lResult = STATUS_INVALID_STATE_TRANSITION; break;
        case IOCTL_RESULT_STOPPED          : lResult = STATUS_INVALID_STATE_TRANSITION; break;
        case IOCTL_RESULT_SYSTEM_ERROR     : lResult = STATUS_UNSUCCESSFUL            ; break;
        case IOCTL_RESULT_TOO_MANY_ADAPTER : lResult = STATUS_TOO_MANY_NODES          ; break;
        case IOCTL_RESULT_TOO_MANY_BUFFER  : lResult = STATUS_TOO_MANY_ADDRESSES      ; break;
        case IOCTL_RESULT_WAIT             : lResult = STATUS_PENDING                 ; break;

        default:
            ASSERT(0 < lIoCtlResult);

            WdfRequestSetInformation(aRequest, lIoCtlResult);
            lResult = STATUS_SUCCESS;
        }

        return lResult;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

// CRITICAL PATH  BufferEvent  1 / Buffer event
void ProcessEvent(void * aContext)
{
    ASSERT(NULL != aContext);

    OpenNetK::Adapter_WDF * lThis = reinterpret_cast<OpenNetK::Adapter_WDF *>(aContext);

    lThis->Event_Process();
}
