
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Linux.cpp

#define __CLASS__ "Adapter_Linux::"

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
#include "CUW.h"
#include "Thread_Functions.h"
#include "Thread_Kernel_CUDA.h"

#include "Adapter_Linux.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aHandle   [DK-;RW-] The handle to the driver
// aDebugLog [-K-;RW-] The debug log
//
// Thread  Apps
Adapter_Linux::Adapter_Linux(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : Adapter_Internal( aHandle, aDebugLog )
    , mModule( NULL )
{
    assert( NULL != aHandle   );
    assert( NULL != aDebugLog );
}

// aBuffers [---;RW-]
//
// Thread  Apps
void Adapter_Linux::Buffers_Allocate( Buffer_Data_Vector * aBuffers )
{
    assert( NULL != aBuffers );

    assert( 0 < mConfig.mBufferQty );

    for ( unsigned int i = 0; i < mConfig.mBufferQty; i ++ )
    {
        Buffer_Data * lBD = Buffer_Allocate();
        assert( NULL != lBD );

        aBuffers->push_back( lBD );
    }
}

// ===== Adapter_Internal ===================================================

// aConnect [---;R--]
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Linux::Connect(IoCtl_Connect_In * aConnect)
{
    // printf( __CLASS__ "Connect( 0x%lx ) - 0x%lx\n", reinterpret_cast< uint64_t >( aConnect ), reinterpret_cast< uint64_t >( aConnect->mSharedMemory ) );

    assert(NULL != aConnect);

    assert(NULL != mHandle);

    mHandle->Control(IOCTL_CONNECT, aConnect, sizeof(IoCtl_Connect_In), NULL, 0);
}

// ===== OpenNet::Adapter ===================================================

Adapter_Linux::~Adapter_Linux()
{
    // printf( __CLASS__ "~Adapter_Linux()\n" );
}

OpenNet::Status Adapter_Linux::Packet_Send(const void * aData, unsigned int aSize_byte)
{
    assert(NULL != mDebugLog);

    if (NULL == aData)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (mInfo.mPacketSize_byte < aSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    mStatistics[OpenNet::ADAPTER_STATS_PACKET_SEND] ++;

    // TODO  OpenNet.Adapter
    //       Create Adapter_Internal::Packet_Send_Internal from there and
    //       move the arguments verification into
    //       Adapter_Internal::Packet_Send. Also move the exception
    //       management into Adapter_Internal::Packet_Send.

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

    // printf( __CLASS__ "Packet_Send - delete [] 0x%lx (lBuffer)\n", reinterpret_cast< uint64_t >( lBuffer ) );
    delete [] lBuffer;

    return lResult;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter_Internal ===================================================

void Adapter_Linux::ResetInputFilter_Internal()
{
    if ( NULL != mModule )
    {
        // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See SetInputFilter_Internal
        CUW_ModuleUnload( mModule );
        mModule = NULL;
    }
}

void Adapter_Linux::SetInputFilter_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert( NULL != mProcessor );
    assert( NULL == mModule    );

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
    assert( NULL != lProcessor );

    // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See ResetInputFilter_Internal
    mModule = lProcessor->Module_Create( aKernel );
    assert( NULL != mModule );
}

Thread * Adapter_Linux::Thread_Prepare_Internal( OpenNet::Kernel * aKernel )
{
    assert( NULL != aKernel );

    assert( NULL != mDebugLog  );
    assert( NULL != mModule    );
    assert( NULL != mProcessor );

    // new ==> delete
    return new Thread_Kernel_CUDA( mProcessor, this, aKernel, mModule, mDebugLog );
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Return  This method returns the address of the allocated buffer.
//
// Exception  KmsLib::Exception *  CODE_NOT_ENOUGH_MEMORY
//                                 See Process_Internal::Buffer_Allocate
// Threads    Apps
Buffer_Data * Adapter_Linux::Buffer_Allocate()
{
    assert( OPEN_NET_BUFFER_QTY >= mBufferCount );
    assert( NULL                != mProcessor   );

    if (OPEN_NET_BUFFER_QTY <= mBufferCount)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY,
            "Too many buffer", NULL, __FILE__, __CLASS__ "Buffer_Allocate", __LINE__, 0);
    }

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
    assert( NULL != lProcessor );

    Buffer_Data * lResult = lProcessor->Buffer_Allocate( mConfig.mPacketSize_byte, mBuffers + mBufferCount );
    assert( NULL != lResult );

    mBufferCount ++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;

    return lResult;
}
