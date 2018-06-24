
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Adapter_WDF.cpp

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
#include <OpenNetK/Interface.h>

#include <OpenNetK/Adapter_WDF.h>

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter_WDF::Init(Adapter * aAdapter, WDFDEVICE aDevice, Hardware_WDF * aHardware_WDF)
    {
        ASSERT(NULL != aAdapter     );
        ASSERT(NULL != aDevice      );
        ASSERT(NULL != aHardware_WDF);

        mAdapter      = aAdapter     ;
        mDevice       = aDevice      ;
        mEvent        = NULL         ;
        mFileObject   = NULL         ;
        mHardware_WDF = aHardware_WDF;

        new (&mZone0) SpinLock_WDF(aDevice);

        mAdapter->Init(&mZone0);
    }

    void Adapter_WDF::FileCleanup(WDFFILEOBJECT aFileObject)
    {
        ASSERT(NULL != aFileObject);

        ASSERT(NULL != mAdapter);

        if (mFileObject == aFileObject)
        {
            mAdapter->Disconnect();

            Disconnect();
        }
    }

    void Adapter_WDF::IoDeviceControl(WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aCode)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mAdapter);

        NTSTATUS lStatus = STATUS_NOT_SUPPORTED;

        Adapter::IoCtlInfo lInfo;

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
                        int lRet = mAdapter->IoCtl(aCode, lIn, static_cast<unsigned int>(aInSize_byte), lOut, static_cast<unsigned int>(aOutSize_byte));

                        ProcessIoCtlResult(lRet);

                        lStatus = ResultToStatus(aRequest, lRet);
                    }
                }
            }
        }

        ASSERT(STATUS_PENDING != lStatus);
        WdfRequestComplete(aRequest, lStatus);
    }

    void Adapter_WDF::IoInCallerContext(WDFREQUEST aRequest)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mDevice);

        WDF_REQUEST_PARAMETERS lParameters;

        WDF_REQUEST_PARAMETERS_INIT(&lParameters);

        WdfRequestGetParameters(aRequest, &lParameters);

        if ( (WdfRequestTypeDeviceControl == lParameters.Type) && (OPEN_NET_IOCTL_CONNECT == lParameters.Parameters.DeviceIoControl.IoControlCode) )
        {
            NTSTATUS lStatus;

            if (NULL == mFileObject)
            {
                OpenNet_Connect * lIn         ;
                size_t            lInSize_byte;

                lStatus = WdfRequestRetrieveInputBuffer(aRequest, sizeof(OpenNet_Connect), reinterpret_cast<PVOID *>(&lIn), &lInSize_byte);
                if (STATUS_SUCCESS == lStatus)
                {
                    ASSERT(NULL                    != lIn         );
                    ASSERT(sizeof(OpenNet_Connect) <= lInSize_byte);

                    lStatus = Connect(lIn, WdfRequestGetFileObject(aRequest));
                }
            }
            else
            {
                lStatus = STATUS_INVALID_STATE_TRANSITION;
            }

            if (STATUS_SUCCESS != lStatus)
            {
                WdfRequestComplete(aRequest, lStatus);
                return;
            }
        }

        WdfDeviceEnqueueRequest(mDevice, aRequest);
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aIn [---;RW-] The input data
    // aFileObject   The file object
    //
    // Return  STATUS_SUCCESS
    //         See Event_Translate
    //         See SharedMemory_Translate
    //
    // Level    PASSIVE
    // Threads  Users
    NTSTATUS Adapter_WDF::Connect(OpenNet_Connect * aIn, WDFFILEOBJECT aFileObject)
    {
        ASSERT(NULL != aIn        );
        ASSERT(NULL != aFileObject);

        // Event_Translate ==> Event_Release  See FileCleanup
        NTSTATUS lResult = Event_Translate(&aIn->mEvent);
        if (STATUS_SUCCESS == lResult)
        {
            ASSERT(NULL != aIn->mEvent);

            // SharedMemory_Translate ==> SharedMemory_Release  See FileCleanup
            lResult = SharedMemory_Translate(&aIn->mSharedMemory);
            if (STATUS_SUCCESS == lResult)
            {
                mFileObject = aFileObject;
            }
            else
            {
                Event_Release();
            }
        }

        return lResult;
    }

    // Level   PASSIVE
    // Thread  Users
    void Adapter_WDF::Disconnect()
    {
        ASSERT(NULL != mFileObject);

        // Event_Translate ==> Event_Release  See IoInCallerContext
        Event_Release();

        // SharedMemory_Translate ==> SharedMemory_Release  See IoInCallerContext
        SharedMemory_Release();

        mFileObject = NULL;
    }

    // Level   PASSIVE
    // Thread  Users
    void Adapter_WDF::Event_Release()
    {
        ASSERT(NULL != mEvent);

        // ObReferenceObjectByHandle ==> ObDereferenceObject  See Event_Translate
        ObDereferenceObject(mEvent);

        mEvent = NULL;
    }

    // Event_Translate ==> Event_Release
    //
    // Level   PASSIVE
    // Thread  Users
    NTSTATUS Adapter_WDF::Event_Translate(uint64_t * aEvent)
    {
        ASSERT(NULL != aEvent);

        ASSERT(NULL == mEvent);

        if (NULL == (*aEvent))
        {
            return STATUS_INVALID_HANDLE;
        }

        if (NULL != mEvent)
        {
            return STATUS_INVALID_STATE_TRANSITION;
        }

        // ObReferenceObjectByHandle ==> ObDereferenceObject  See Event_Release
        NTSTATUS lResult = ObReferenceObjectByHandle(reinterpret_cast<HANDLE>(*aEvent), SYNCHRONIZE, *ExEventObjectType, UserMode, reinterpret_cast<PVOID *>(&mEvent), NULL);
        if (STATUS_SUCCESS == lResult)
        {
            ASSERT(NULL != mEvent);

            (*aEvent) = reinterpret_cast<uint64_t>(mEvent);
        }

        return lResult;
    }

    // Level   DISPATCH
    // Thread  Queue
    void Adapter_WDF::ProcessIoCtlResult(int aResult)
    {
        ASSERT(NULL != mHardware_WDF);

        switch (aResult)
        {
        case Adapter::IOCTL_RESULT_INVALID_SYSTEM_ID:
        case Adapter::IOCTL_RESULT_TOO_MANY_ADAPTER :
            Disconnect();
            break;

        case Adapter::IOCTL_RESULT_PROCESSING_NEEDED:
            mHardware_WDF->TrigProcess2();
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
        mSharedMemory_MDL = IoAllocateMdl(*aSharedMemory, OPEN_NET_SHARED_MEMORY_SIZE_byte, FALSE, FALSE, NULL);
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

    NTSTATUS Adapter_WDF::ResultToStatus(WDFREQUEST aRequest, int aResult)
    {
        ASSERT(NULL != aRequest);

        NTSTATUS lResult;

        switch (aResult)
        {
        case Adapter::IOCTL_RESULT_OK               :
        case Adapter::IOCTL_RESULT_PROCESSING_NEEDED:
            lResult = STATUS_SUCCESS;
            break;

        case Adapter::IOCTL_RESULT_ERROR           : lResult = STATUS_UNSUCCESSFUL      ; break;
        case Adapter::IOCTL_RESULT_TOO_MANY_ADAPTER: lResult = STATUS_TOO_MANY_NODES    ; break;
        case Adapter::IOCTL_RESULT_TOO_MANY_BUFFER : lResult = STATUS_TOO_MANY_ADDRESSES; break;

        default:
            WdfRequestSetInformation(aRequest, aResult);
            lResult = STATUS_SUCCESS;
        }

        return lResult;
    }

}
