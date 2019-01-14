
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_CUDA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "Processor_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aDevice
// aDebugLog [-K-;RW-]
//
// Exception  KmsLib::Exception *  See InitInfo
// Threads  Apps
Processor_CUDA::Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog )
    : Processor_Internal( aDebugLog )
{
    assert( NULL != aDebugLog );

    InitInfo();
}

// ===== Processor_Internal =================================================

Thread_Functions * Processor_CUDA::Thread_Get()
{
    assert(NULL != mDebugLog);

    if (NULL == mThread)
    {
        mThread = new Thread_Functions_CUDA(this, mConfig.mFlags.mProfilingEnabled, mDebugLog);
        assert(NULL != mThread);
    }

    return mThread;
}

// ===== OpenNet::Processor =================================================

Processor_CUDA::~Processor_CUDA()
{
}

void * Processor_CUDA::GetContext()
{
    // TODO  Dev
    return NULL;
}

void * Processor_CUDA::GetDeviceId()
{
    // TODO  Dev
    return NULL;
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Processor_CUDA::InitInfo()
{
}
