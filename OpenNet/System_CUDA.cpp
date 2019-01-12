
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_CUDA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>

// ===== System =============================================================
#include <sys/eventfd.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "Adapter_Linux.h"
#include "CUDAW.h"
#include "Processor_CUDA.h"

#include "System_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

System_CUDA::System_CUDA()
{
    assert( NULL == mConnect.mSharedMemory );

    mInfo.mSystemId = getpid();
    assert(0 != mInfo.mSystemId);

    mConnect.mSystemId = mInfo.mSystemId;

    FindAdapters  ();
    FindProcessors();

    // eventfd ==> close  See the destructor
    int lEvent = eventfd( 0, EFD_CLOEXEC );
    assert( 0 <= lEvent );

    mConnect.mEvent = lEvent;

    // valloc ==> free  See the descructor
    mConnect.mSharedMemory = valloc( SHARED_MEMORY_SIZE_byte );
    assert( NULL != mConnect.mSharedMemory );
}

System_CUDA::~System_CUDA()
{
    assert(    0 <= mConnect.mEvent        );
    assert( NULL != mConnect.mSharedMemory );

    // eventfd ==> close  See the constructor
    int lRet = close( mConnect.mEvent );
    assert( 0 == lRet );
    (void)( lRet );

    // valloc ==> free  See the constructor
    free( mConnect.mSharedMemory );
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
void System_CUDA::FindAdapters()
{
    mDebugLog.Log( "System_CUDA::FindAdapters()" );

    for (unsigned int lIndex = 0;; lIndex++)
    {
        // new ==> delete  See Adapter_Internal::~Adapter_Internal
        KmsLib::DriverHandle * lHandle = new KmsLib::DriverHandle();
        assert(NULL != lHandle);

        try
        {
            char lName[16];

            sprintf_s(lName, "/dev/OpenNet%d", lIndex);

            lHandle->Connect(lName, O_RDWR);
        }
        catch (KmsLib::Exception * eE)
        {
            (void)(eE);

            delete lHandle;
            break;
        }

        // new ==> delete  See ~System_Internal
        Adapter_Internal * lAdapter = new Adapter_Linux(lHandle, &mDebugLog);
        assert(NULL != lAdapter);

        mAdapters.push_back(lAdapter);
    }
}

// Exception  KmsLib::Exception *  See OCLW_GetDeviceIDs
// Threads  Apps
void System_CUDA::FindProcessors()
{
    mDebugLog.Log( "System_CUDA::FindProcessors()" );

    int lCount;

    CUDAW_GetDeviceCount( & lCount );

    for ( int i = 0; i < lCount; i ++ )
    {
        mProcessors.push_back( new Processor_CUDA( i, & mDebugLog ) );
    }
}
