
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Windows.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"

// ===== OpenNet ============================================================
#include "OCLW.h"
#include "Processor_OpenCL.h"
#include "Thread_Kernel_OpenCL.h"

#include "Adapter_Windows.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aHandle   [DK-;RW-] The handle to the driver
// aDebugLog [-K-;RW-] The address of the DebugLog instance to use
//
// Exception  KmsLib::Exception *
// Threads    Apps
Adapter_Windows::Adapter_Windows(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : Adapter_Internal( aHandle, aDebugLog )
    , mProgram(NULL)
{
}

// aCommandQueue [---;RW-]
// aKernel       [---;R--]
// aBuffers      [---;RW-] The caller is responsible to release the
//                         Buffer_Data instances added to this queue when
//                         they are no longer needed.
//
// Exception  KmsLib::Exception *  See Adapter_Internal::Buffer_Allocate
// Thread     Apps
void Adapter_Windows::Buffers_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Data_Vector * aBuffers)
{
    assert(NULL != aCommandQueue);
    assert(NULL != aKernel      );
    assert(NULL != aBuffers     );

    assert(0 < mConfig.mBufferQty);

    for (unsigned int i = 0; i < mConfig.mBufferQty; i++)
    {
        aBuffers->push_back(Buffer_Allocate(aCommandQueue, aKernel));
    }
}

// ===== Adapter_Internal ===================================================

// aKernel [---;RW-]
//
// Return  This method returns the address of the newly created Thread
//         instance. The caller is responsible for deleting this instance.
//
// Thread  Apps
Thread * Adapter_Windows::Thread_Prepare_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert(NULL != mDebugLog );
    assert(NULL != mProcessor);
    assert(NULL != mProgram  );

    // new ==> delete
    return new Thread_Kernel_OpenCL(mProcessor, this, aKernel, mProgram, mDebugLog);
}

// ===== OpenNet::Adapter ===================================================

Adapter_Windows::~Adapter_Windows()
{
    try
    {
        if (NULL != mProgram)
        {
            OCLW_ReleaseProgram(mProgram);
        }
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter_Internal ===================================================

void Adapter_Windows::ResetInputFilter_Internal()
{
    if (NULL != mProgram)
    {
        OCLW_ReleaseProgram(mProgram);
        mProgram = NULL;
    }
}

void Adapter_Windows::SetInputFilter_Internal(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert(NULL != mProcessor);
    assert(NULL == mProgram  );

    Processor_OpenCL * lProcessor = dynamic_cast<Processor_OpenCL *>(mProcessor);
    assert(NULL != lProcessor);

    mProgram = lProcessor->Program_Create(aKernel);
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aCommandQueue [---;RW-]
// aKernel       [---;R--]
//
// Return  This method returns the address of the allocated buffer.
//
// Exception  KmsLib::Exception *  CODE_NOT_ENOUGH_MEMORY
//                                 See Process_Internal::Buffer_Allocate
// Threads    Apps
Buffer_Data * Adapter_Windows::Buffer_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel)
{
    assert(NULL != aCommandQueue);
    assert(NULL != aKernel      );

    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(NULL                != mProcessor  );

    if (OPEN_NET_BUFFER_QTY <= mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY,
            "Too many buffer", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    Processor_OpenCL * lProcessor = dynamic_cast<Processor_OpenCL *>(mProcessor);
    assert(NULL != lProcessor);

    Buffer_Data * lResult = lProcessor->Buffer_Allocate(mConfig.mPacketSize_byte, aCommandQueue, aKernel, mBuffers + mBufferCount);

    mBufferCount++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;

    return lResult;
}
