
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_Internal.cpp

#define __CLASS__ "Processor_Internal::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "Thread_Functions.h"

#include "Processor_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aDebugLog [-K-;RW-]
//
// Threads  Apps
Processor_Internal::Processor_Internal( KmsLib::DebugLog * aDebugLog )
    : mDebugLog( aDebugLog )
    , mThread  ( NULL      )
{
    assert( NULL != aDebugLog );

    memset(&mInfo, 0, sizeof(mInfo));
}

// Return  This methode return the address of the internal thread instance.
Thread * Processor_Internal::Thread_Prepare()
{
    // printf( __CLASS__ "Thread_Prepare()\n" );

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

// ===== OpenNet::Processor =================================================

Processor_Internal::~Processor_Internal()
{
}

OpenNet::Status Processor_Internal::GetConfig(Config * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetConfig", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(Config));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Processor_Internal::GetInfo(Info * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetInfo", __LINE__);
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
        mDebugLog->Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    memcpy(&mConfig, &aConfig, sizeof(Config));

    return OpenNet::STATUS_OK;
}

OpenNet::UserBuffer * Processor_Internal::AllocateUserBuffer(unsigned int aSize_byte)
{
    if (0 >= aSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "AllocateUserBuffer", __LINE__);
        return NULL;
    }

    OpenNet::UserBuffer * lResult;

    try
    {
        lResult = AllocateUserBuffer_Internal(aSize_byte);
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "AllocateUserBuffer", __LINE__);
        mDebugLog->Log(eE);

        lResult = NULL;
    }

    return lResult;
}


OpenNet::Status Processor_Internal::Display(FILE * aOut) const
{
    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Display", __LINE__);
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
        mDebugLog->Log(__FILE__, __CLASS__ "GetStatistics", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 < aOutSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetStatistics", __LINE__);
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
