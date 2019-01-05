
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All right reserved.
// Product    OpenNet
// File       ONK_Lib/Adapter_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Hardware_Linux.h>

#include <OpenNetK/Adapter_Linux.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"

// ===== ONK_Lib ============================================================
#include "IoCtl.h"

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter_Linux::Init( Adapter * aAdapter, Hardware_Linux * aHardware_Linux )
    {
        ASSERT( NULL != aAdapter        );
        ASSERT( NULL != aHardware_Linux );

        mAdapter        = aAdapter       ;
        mHardware_Linux = aHardware_Linux;

        new ( & mZone0 ) SpinLock_Linux();

        mAdapter->Init( & mZone0 );
    }

    void Adapter_Linux::FileCleanup()
    {
        ASSERT( NULL != mAdapter );

        // TODO  Dev  if ( mFileObject == aFileObject )
        {
            mAdapter->Disconnect();

            Disconnect();
        }
    }

    int Adapter_Linux::IoDeviceControl( void * aInOut, size_t aSize_byte, unsigned int aCode )
    {
        ASSERT( NULL != mAdapter );

        IoCtl_Info lInfo;

        if ( ! mAdapter->IoCtl_GetInfo( aCode, & lInfo ) )
        {
            return ( - __LINE__ );
        }

        if ( ( aSize_byte < lInfo.mIn_MinSize_byte ) || ( aSize_byte < lInfo.mOut_MinSize_byte ) )
        {
            return ( - __LINE__ );
        }

        int lRet = mAdapter->IoCtl( aCode, aInOut, static_cast< unsigned int >( aSize_byte ), aInOut, static_cast< unsigned int >( aSize_byte ) );

        ProcessIoCtlResult( lRet );

        return ResultToStatus( lRet );
    }

    void Adapter_Linux::IoInCallerContext()
    {
        /* TODO  Dev           IoCtl_Connect_In * lIn         ;
                size_t             lInSize_byte;

                    ASSERT(NULL                     != lIn         );
                    ASSERT(sizeof(IoCtl_Connect_In) <= lInSize_byte);

                    lStatus = Connect(lIn, WdfRequestGetFileObject(aRequest)); */
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
    int Adapter_Linux::Connect( void * aIn )
    {
        ASSERT( NULL != aIn );

        IoCtl_Connect_In * lIn = reinterpret_cast< IoCtl_Connect_In * >( aIn );

        return 0;
    }

    // Level   PASSIVE
    // Thread  Users
    void Adapter_Linux::Disconnect()
    {
        // Event_Translate ==> Event_Release  See IoInCallerContext
        Event_Release();

        // SharedMemory_Translate ==> SharedMemory_Release  See IoInCallerContext
        SharedMemory_Release();
    }

    // Level   PASSIVE
    // Thread  Users
    void Adapter_Linux::Event_Release()
    {
    }

    // Event_Translate ==> Event_Release
    //
    // Level   PASSIVE
    // Thread  Users
    int Adapter_Linux::Event_Translate(uint64_t * aEvent)
    {
        ASSERT(NULL != aEvent);

        if (NULL == (*aEvent))
        {
            return ( - __LINE__ );
        }

        return 0;
    }

    // Level   DISPATCH
    // Thread  Queue
    void Adapter_Linux::ProcessIoCtlResult(int aIoCtlResult)
    {
        ASSERT( NULL != mHardware_Linux );

        IoCtl_Result lIoCtlResult = static_cast< IoCtl_Result >( aIoCtlResult );

        switch (lIoCtlResult)
        {
        case IOCTL_RESULT_INVALID_SYSTEM_ID:
        case IOCTL_RESULT_TOO_MANY_ADAPTER :
            Disconnect();
            break;

        case IOCTL_RESULT_PROCESSING_NEEDED:
            mHardware_Linux->TrigProcess2();
            break;
        }
    }

    // Level   PASSIVE
    // Thread  Users
    int Adapter_Linux::SharedMemory_ProbeAndLock()
    {
        return 0;
    }

    // SharedMemory_Translate ==> SharedMemory_Release
    //
    // Level   PASSIVE
    // Thread  Users
    void Adapter_Linux::SharedMemory_Release()
    {
    }

    // aSharedMemory [DK-;RW-]
    //
    // SharedMemory_Translate ==> SharedMemory_Release
    //
    // Level    PASSIVE
    // Threads  Users
    int Adapter_Linux::SharedMemory_Translate(void ** aSharedMemory)
    {
        ASSERT( NULL != aSharedMemory );

        if (NULL == ( * aSharedMemory ) )
        {
            return ( - __LINE__ );
        }

        return 0;
    }

    int Adapter_Linux::ResultToStatus( int aIoCtlResult )
    {
        IoCtl_Result lIoCtlResult = static_cast< IoCtl_Result >( aIoCtlResult );

        int lResult;

        switch ( lIoCtlResult )
        {
        case IOCTL_RESULT_OK                :
        case IOCTL_RESULT_PROCESSING_NEEDED :
            lResult = 0;
            break;

        case IOCTL_RESULT_ERROR            : lResult = ( - __LINE__ ); break;
        case IOCTL_RESULT_TOO_MANY_ADAPTER : lResult = ( - __LINE__ ); break;
        case IOCTL_RESULT_TOO_MANY_BUFFER  : lResult = ( - __LINE__ ); break;

        default:
            lResult = aIoCtlResult;
        }

        return lResult;
    }

}
