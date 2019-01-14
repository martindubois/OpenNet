
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNet/Function.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"

// ===== OpenNet ============================================================
#include "Thread_Functions.h"

#include "Adapter_Linux.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aHandle   [DK-;RW-] The handle to the driver
// aDebugLog [-K-;RW-] The debug log
//
// Thread  Apps
Adapter_Linux::Adapter_Linux(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : Adapter_Internal( aHandle, aDebugLog )
{
    assert( NULL != aHandle   );
    assert( NULL != aDebugLog );
}

// ===== Adapter_Internal ===================================================

// aConnect [---;R--]
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Linux::Connect(IoCtl_Connect_In * aConnect)
{
    assert(NULL != aConnect);

    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    try
    {
        mHandle->Control(IOCTL_CONNECT, aConnect, sizeof(IoCtl_Connect_In), NULL, 0);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        throw;
    }
}

// ===== OpenNet::Adapter ===================================================

Adapter_Linux::~Adapter_Linux()
{
}

OpenNet::Status Adapter_Linux::Packet_Send(const void * aData, unsigned int aSize_byte)
{
    assert(NULL != mDebugLog);

    if (NULL == aData)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (mInfo.mPacketSize_byte < aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    mStatistics[OpenNet::ADAPTER_STATS_PACKET_SEND] ++;

    unsigned char * lBuffer = new unsigned char [ sizeof( IoCtl_Packet_Send_Ex_In ) + aSize_byte ];
    assert( NULL != lBuffer );

    IoCtl_Packet_Send_Ex_In * lIn = reinterpret_cast< IoCtl_Packet_Send_Ex_In * >( lBuffer );

    memset( lIn, 0, sizeof( IoCtl_Packet_Send_Ex_In ) );
    memcpy( lIn + 1, aData, aSize_byte );

    lIn->mRepeatCount =          1;
    lIn->mSize_byte   = aSize_byte;

    OpenNet::Status lResult;

    try
    {
        Packet_Send_Ex( lIn );
        lResult = OpenNet::STATUS_OK;
    }
    catch ( KmsLib::Exception * eE )
    {
        mDebugLog->Log( eE );
        lResult = OpenNet::STATUS_EXCEPTION;
    }

    delete [] lBuffer;

    return lResult;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter_Internal ===================================================

OpenNet::Status Adapter_Linux::ResetInputFilter_Internal()
{
    // TODO  Dev

    return OpenNet::STATUS_OK;
}

void Adapter_Linux::SetInputFilter_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    // TODO  Dev
}

Thread * Adapter_Linux::Thread_Prepare_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    // TODO  Dev
    return NULL;
}
