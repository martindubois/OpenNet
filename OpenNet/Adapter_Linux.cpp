
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Linux.cpp

#define __CLASS__ "Adapter_Linux::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNet/Function.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
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
// Threads  Apps
Adapter_Linux::Adapter_Linux(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : Adapter_Internal( aHandle, aDebugLog )
    , mModule( NULL )
{
    assert( NULL != aHandle   );
    assert( NULL != aDebugLog );
}

// aProfiling
// aBuffers [---;RW-]
//
// Thread  Apps
void Adapter_Linux::Buffers_Allocate( bool aProfiling, Buffer_Internal_Vector * aBuffers )
{
    assert( NULL != aBuffers );

    if (0 < mConfig.mBufferQty)
    {
        memset(&mBuffers, 0, sizeof(mBuffers));

        for (unsigned int i = 0; i < mConfig.mBufferQty; i++)
        {
            Buffer_Inernal * lBuffer = Buffer_Allocate(aProfiling);
            assert(NULL != lBuffer);

            aBuffers->push_back(lBuffer);
        }
    }
}

// ===== OpenNet::Adapter ===================================================

Adapter_Linux::~Adapter_Linux()
{
    // printf( __CLASS__ "~Adapter_Linux()\n" );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter_Internal ===================================================

void Adapter_Linux::ResetInputFilter_Internal()
{
    if ( NULL != mModule )
    {
        assert( NULL != mProcessor );

        Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
        assert( NULL != lProcessor );

        lProcessor->SetContext();

        // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See SetInputFilter_Internal
        CUW_ModuleUnload( mModule );
        mModule = NULL;
    }
}

void Adapter_Linux::SetInputFilter_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert( ADAPTER_NO_UNKNOWN != mConnect_Out.mAdapterNo );
    assert( NULL != mProcessor );
    assert( NULL == mModule    );

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( mProcessor );
    assert( NULL != lProcessor );

    // Processor_CUDA::Module_Create ==> CUW_ModuleUnload  See ResetInputFilter_Internal
    mModule = lProcessor->Module_Create( aKernel, mConnect_Out.mAdapterNo );
    assert( NULL != mModule );
}

void Adapter_Linux::Stop_Internal()
{
    assert(NULL != mHandle);

    mHandle->Control(IOCTL_EVENT_WAIT_CANCEL, NULL, 0, NULL, 0);
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

// aProfiling
//
// Return  This method returns the address of the allocated buffer.
//
// Exception  KmsLib::Exception *  CODE_NOT_ENOUGH_MEMORY
//                                 See Process_CUDA::Buffer_Allocate
// Threads    Apps
Buffer_Internal * Adapter_Linux::Buffer_Allocate( bool aProfiling )
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

    Buffer_Internal * lResult = lProcessor->Buffer_Allocate( aProfiling, mConfig.mPacketSize_byte, mBuffers + mBufferCount );
    assert( NULL != lResult );

    mBufferCount ++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;

    return lResult;
}
