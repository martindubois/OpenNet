
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Windows.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
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
void Adapter_Internal::Buffers_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Data_Vector * aBuffers)
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

// aConnect [---;R--]
//
// Exception  KmsLib::Exception *  CODE_TIMEOUT
//                                 See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Windows::Connect(IoCtl_Connect_In * aConnect)
{
    assert(NULL != aConnect);

    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    HANDLE lEvent = reinterpret_cast<HANDLE>(aConnect->mEvent);
    assert(NULL != lEvent);

    DWORD lRet = WaitForSingleObject(lEvent, 60000);
    if (WAIT_OBJECT_0 != lRet)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
            "WaitForSingleObject( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet);
    }

    try
    {
        mHandle->Control(IOCTL_CONNECT, aConnect, sizeof(IoCtl_Connect_In), NULL, 0);

        SetEvent(lEvent);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        SetEvent(lEvent);
 
        throw;
    }
}

// Return  This method returns the address of the newly created Thread
//         instance. The caller is responsible for deleting this instance.
//
// Thread  Apps
Thread * Adapter_Windows::Thread_Prepare()
{
    assert(NULL != mDebugLog  );

    if (NULL != mSourceCode)
    {
        assert(NULL != mProcessor);

        OpenNet::Kernel * lKernel = dynamic_cast<OpenNet::Kernel *>(mSourceCode);
        if (NULL != lKernel)
        {
            assert(NULL != mProgram);

            // new ==> delete
            return new Thread_Kernel(mProcessor, this, lKernel, mProgram, mDebugLog);
        }

        OpenNet::Function * lFunction = dynamic_cast<OpenNet::Function *>(mSourceCode);
        assert(NULL != lFunction);

        Thread_Functions * lThread = mProcessor->Thread_Get();
        assert(NULL != lThread);

        lThread->AddAdapter(this, *lFunction);
    }

    return NULL;
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

OpenNet::Status Adapter_Windows::Packet_Send(const void * aData, unsigned int aSize_byte)
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

    return Control(IOCTL_PACKET_SEND, aData, aSize_byte, NULL, 0);
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
Buffer_Data * Adapter_Internal::Buffer_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel)
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

    Buffer_Data * lResult = mProcessor->Buffer_Allocate(mConfig.mPacketSize_byte, aCommandQueue, aKernel, mBuffers + mBufferCount);

    mBufferCount++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;

    return lResult;
}
