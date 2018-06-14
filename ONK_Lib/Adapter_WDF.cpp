
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
#include <OpenNetK/Interface.h>

#include <OpenNetK/Adapter_WDF.h>

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter_WDF::Init(Adapter * aAdapter, WDFDEVICE aDevice)
    {
        ASSERT(NULL != aAdapter);
        ASSERT(NULL != aDevice );

        mAdapter = aAdapter;
        mDevice  = aDevice ;

        mAdapter->Init();
    }

    void Adapter_WDF::IoDeviceControl(WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aCode)
    {
        ASSERT(NULL != aRequest);

        ASSERT(NULL != mAdapter);

        unsigned int lInSizeMin_byte  = mAdapter->IoCtl_InSize_GetMin (aCode);
        unsigned int lOutSizeMin_byte = mAdapter->IoCtl_OutSize_GetMin(aCode);

        NTSTATUS lStatus;

        if ((aInSize_byte < lInSizeMin_byte) || (aOutSize_byte < lOutSizeMin_byte))
        {
            lStatus = STATUS_INVALID_BUFFER_SIZE;
        }
        else
        {
            void * lIn = NULL;

            if (0 < aInSize_byte)
            {
                lStatus = WdfRequestRetrieveInputBuffer(aRequest, lInSizeMin_byte, &lIn, NULL);
            }
            else
            {
                lStatus = STATUS_SUCCESS;
            }

            if (STATUS_SUCCESS == lStatus)
            {
                void * lOut = NULL;

                if (0 < aOutSize_byte)
                {
                    lStatus = WdfRequestRetrieveOutputBuffer(aRequest, lOutSizeMin_byte, &lOut, NULL);
                }

                if (STATUS_SUCCESS == lStatus)
                {
                    int lRet = mAdapter->IoCtl(aCode, lIn, static_cast<unsigned int>(aInSize_byte), lOut, static_cast<unsigned int>(aOutSize_byte));
                    if (0 <= lRet)
                    {
                        WdfRequestSetInformation(aRequest, lRet);
                    }
                    else
                    {
                        lStatus = STATUS_UNSUCCESSFUL;
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
            OpenNet_Connect * lIn         ;
            size_t            lInSize_byte;

            NTSTATUS lStatus = WdfRequestRetrieveInputBuffer(aRequest, sizeof(OpenNet_Connect), reinterpret_cast<PVOID *>(&lIn), &lInSize_byte);
            if (STATUS_SUCCESS == lStatus)
            {
                ASSERT(NULL                    != lIn         );
                ASSERT(sizeof(OpenNet_Connect) <= lInSize_byte);

                PKEVENT lEvent;

                lStatus = ObReferenceObjectByHandle(reinterpret_cast<HANDLE>(lIn->mEvent), GENERIC_ALL, *ExEventObjectType, UserMode, reinterpret_cast<PVOID *>(&lEvent), NULL);
                if (STATUS_SUCCESS == lStatus)
                {
                    // TODO  Test
                    ASSERT(NULL != lEvent);

                    lIn->mEvent = reinterpret_cast<uint64_t>(lEvent);
                }
            }

            if (STATUS_SUCCESS != lStatus)
            {
                WdfRequestComplete(aRequest, lStatus);
                return;
            }
        }

        WdfDeviceEnqueueRequest(mDevice, aRequest);
    }

}
