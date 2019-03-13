
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_CUDA.cpp

#define __CLASS__ "System_CUDA::"

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
#include "CUW.h"
#include "Processor_CUDA.h"

#include "System_CUDA.h"

// Static variable
/////////////////////////////////////////////////////////////////////////////

static unsigned int sCounter;

// Public
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
System_CUDA::System_CUDA()
{
    // printf( __CLASS__ "System_CUDA()\n" );

    assert( NULL == mConnect_In.mSharedMemory );

    mInfo.mSystemId = getpid();
    assert(0 != mInfo.mSystemId);

    mConnect_In.mSystemId = mInfo.mSystemId;

    FindAdapters  ();

    if ( 0 == sCounter )
    {
        try
        {
            CUW_Init( 0 );
        }
        catch ( KmsLib::Exception * eE )
        {
            mDebugLog.Log( __FILE__, __CLASS__ "System_CUDA", __LINE__ );
            mDebugLog.Log( eE );
        }
    }

    sCounter ++;

    FindProcessors();

    // valloc ==> free  See the descructor
    mConnect_In.mSharedMemory = valloc( SHARED_MEMORY_SIZE_byte );
    assert( NULL != mConnect_In.mSharedMemory );

    memset( mConnect_In.mSharedMemory, 0, SHARED_MEMORY_SIZE_byte );
}

// ===== OpenNet::~System ===================================================

System_CUDA::~System_CUDA()
{
    // printf( __CLASS__ "~System_CUDA()\n" );

    assert( NULL != mConnect_In.mSharedMemory );

    assert( 0 < sCounter );

    Cleanup();

    // valloc ==> free  See the constructor
    free( mConnect_In.mSharedMemory );

    sCounter --;

    if ( 0 == sCounter )
    {
        CUW_Check();
    }
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
void System_CUDA::FindAdapters()
{
    mDebugLog.Log( __CLASS__ "FindAdapters()" );

    for (unsigned int lIndex = 0;; lIndex++)
    {
        // new ==> delete  See Adapter_Internal::~Adapter_Linux
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

            // printf( __CLASS__ "FindAdapters - delete 0x%lx (lHandle)\n", reinterpret_cast< uint64_t >( lHandle ) );

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
// Threads    Apps
void System_CUDA::FindProcessors()
{
    mDebugLog.Log( __CLASS__ "FindProcessors()" );

    int lCount;

    try
    {
        CUW_DeviceGetCount( & lCount );
    }
    catch ( KmsLib::Exception * eE )
    {
            mDebugLog.Log( __FILE__, __CLASS__ "FindProcessors", __LINE__ );
            mDebugLog.Log( eE );

            return;
    }

    for ( int i = 0; i < lCount; i ++ )
    {
        try
        {
            mProcessors.push_back( new Processor_CUDA( i, & mDebugLog ) );
        }
        catch ( KmsLib::Exception * eE )
        {
            mDebugLog.Log( __FILE__, __CLASS__ "FindProcessors", __LINE__ );
            mDebugLog.Log( eE );
        }
    }
}
