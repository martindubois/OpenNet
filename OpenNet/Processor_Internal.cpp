
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Types.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "Buffer_Data.h"
#include "Constants.h"
#include "Thread_Functions.h"

#include "Processor_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aDebugLog [-K-;RW-]
//
// Threads  Apps
Processor_Internal::Processor_Internal( KmsLib::DebugLog * aDebugLog )
    : mDebugLog( aDebugLog )
{
    assert( NULL != aDebugLog );

    memset(&mInfo, 0, sizeof(mInfo));
}

// Threads  Apps
Processor_Internal::~Processor_Internal()
{
}

Thread_Functions * Processor_Internal::Thread_Get()
{
    assert(NULL != mDebugLog);

    if (NULL == mThread)
    {
        mThread = new Thread_Functions(this, mConfig.mFlags.mProfilingEnabled, mDebugLog);
        assert(NULL != mThread);
    }

    return mThread;
}

Thread * Processor_Internal::Thread_Prepare()
{
    if (NULL != mThread)
    {
        mThread->AddDispatchCode();
    }

    return mThread;
}

void Processor_Internal::Thread_Release()
{
    assert(NULL != mThread);

    mThread = NULL;
}

#ifdef _KMS_WINDOWS_


#endif

// ===== OpenNet::Processor =================================================

OpenNet::Status Processor_Internal::GetConfig(Config * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(Config));

    return OpenNet::STATUS_OK;
}

void * Processor_Internal::GetContext()
{
    #ifdef _KMS_WINDOWS_

        assert(NULL != mContext);

        return mContext;

    #endif
}

void * Processor_Internal::GetDeviceId()
{
    #ifdef _KMS_WINDOWS_

        assert(NULL != mDevice);

        return mDevice;

    #endif
}

OpenNet::Status Processor_Internal::GetInfo(Info * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(Info));

    return OpenNet::STATUS_OK;
}

const char * Processor_Internal::GetName() const
{
    return mInfo.mName;
}

OpenNet::Status Processor_Internal::SetConfig(const Config & aConfig)
{
    assert(NULL != mDebugLog);

    if (NULL == (&aConfig))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    memcpy(&mConfig, &aConfig, sizeof(Config));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Processor_Internal::Display(FILE * aOut) const
{
    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Processor :\n");

    return Processor::Display(mInfo, aOut);
}

// ===== OpenNet::StatisticsProvider ========================================

OpenNet::Status Processor_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 < aOutSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_BUFFER_TOO_SMALL;
    }

    // TODO  OpenNet::Process_Internal
    //       Low (Feature) - Statistics

    if (aReset)
    {
        return ResetStatistics();
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Processor_Internal::ResetStatistics()
{
    // TODO  OpenNet::Process_Internal
    //       Low (Feature) - Statistics

    return OpenNet::STATUS_OK;
}
