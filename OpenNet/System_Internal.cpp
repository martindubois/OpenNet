
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_Internal.cpp

#define __CLASS__ "System_Internal::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <string.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/DriverHandle.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"
#include "Constants.h"
#include "Processor_Internal.h"
#include "Thread.h"

#include "System_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const char * STATE_NAMES[System_Internal::STATE_QTY] =
{
    "IDLE"   ,
    "RUNNING",
};

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE);

static void SendLoopBackPackets(void * aThis, Adapter_Internal * aAdapter);

// Public
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  See FindPlatform
//                                 See FindProcessors
// Threads  Apps
System_Internal::System_Internal()
    : mDebugLog( DEBUG_LOG_FOLDER, "System" )
    , mState   (STATE_IDLE)
{
    mDebugLog.Log( "System_Internal()" );

    memset(&mConfig , 0, sizeof(mConfig ));
    memset(&mConnect, 0, sizeof(mConnect));
    memset(&mInfo   , 0, sizeof(mInfo   ));

    mConfig.mPacketSize_byte = PACKET_SIZE_MAX_byte;

    mDebugLog.Log( "System_Internal::System_Internal - OK" );
}

// ===== OpenNet::System ====================================================

System_Internal::~System_Internal()
{
}

OpenNet::Status System_Internal::GetConfig(Config * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(Config));

    return OpenNet::STATUS_OK;
}

OpenNet::Status System_Internal::GetInfo(Info * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(mInfo));

    return OpenNet::STATUS_OK;
}

OpenNet::Status System_Internal::SetConfig(const Config & aConfig)
{
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

    if (NULL == (&aConfig))
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    OpenNet::Status lResult = Config_Validate(aConfig);
    if (OpenNet::STATUS_OK != lResult)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return lResult;
    }

    switch (mState)
    {
    case STATE_IDLE :
        break;

    case STATE_RUNNING :
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_RUNNING;

    default: assert(false);
    }

    return Config_Apply(aConfig);
}

OpenNet::Status System_Internal::Adapter_Connect(OpenNet::Adapter * aAdapter)
{
    OpenNet::Status lResult = Adapter_Validate(aAdapter);
    if (OpenNet::STATUS_OK != lResult)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return lResult;
    }

    if (aAdapter->IsConnected())
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_ADAPTER_ALREADY_CONNECTED;
    }

    switch (mState)
    {
    case STATE_IDLE :
        break;

    case STATE_RUNNING :
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_RUNNING;

    default: assert(false);
    }

    try
    {
        Adapter_Internal * lAdapter = dynamic_cast<Adapter_Internal *>(aAdapter);
        assert(NULL != lAdapter);

        unsigned int lPacketSize_byte = lAdapter->GetPacketSize();

        if (lPacketSize_byte < mConfig.mPacketSize_byte)
        {
            SetPacketSize(lPacketSize_byte);
            lResult = OpenNet::STATUS_OK;
        }
        else if (lPacketSize_byte > mConfig.mPacketSize_byte)
        {
            lAdapter->SetPacketSize(mConfig.mPacketSize_byte);
        }

        if (OpenNet::STATUS_OK == lResult)
        {
            lAdapter->Connect(&mConnect);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        lResult = ExceptionToStatus(eE);
    }

    return lResult;
}

OpenNet::Adapter * System_Internal::Adapter_Get(unsigned int aIndex)
{
    if (mAdapters.size() <= aIndex)
    {
        return NULL;
    }

    return mAdapters[aIndex];
}

OpenNet::Adapter * System_Internal::Adapter_Get(const unsigned char * aAddress, const unsigned char * aMask, const unsigned char * aMaskDiff)
{
    if ((NULL == aAddress) || (NULL == aMask) || (NULL == aMaskDiff))
    {
        return NULL;
    }

    OpenNet::Adapter * lResult = NULL;

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        bool                   lDiffNeeded = false;
        bool                   lDiffOk     = false;
        OpenNet::Adapter::Info lInfo              ;

        lResult = mAdapters[i];
        assert(NULL != lResult);

        OpenNet::Status lStatus = mAdapters[i]->GetInfo(&lInfo);
        if (OpenNet::STATUS_OK == lStatus)
        {
            for (unsigned int j = 0; j < 6; j++)
            {
                if ((aAddress[j] & aMask[j]) != (lInfo.mEthernetAddress.mAddress[j] & aMask[j]))
                {
                    lResult = NULL;
                    break;
                }

                if (0 != aMaskDiff[j])
                {
                    lDiffNeeded = true;
                    if ((aAddress[j] & aMaskDiff[j]) != (lInfo.mEthernetAddress.mAddress[j] & aMaskDiff[j]))
                    {
                        lDiffOk = true;
                    }
                }
            }

            if ((NULL != lResult) && ((!lDiffNeeded) || lDiffOk))
            {
                break;
            }
        }

        lResult = NULL;
    }

    return lResult;
}

unsigned int System_Internal::Adapter_GetCount() const
{
    return static_cast<unsigned int>(mAdapters.size());
}

OpenNet::Status System_Internal::Display(FILE * aOut)
{
    assert(0 != mConnect.mSystemId);

    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "System :\n");
    fprintf(aOut, "  %zu Adapters\n"    , mAdapters  .size() );
    fprintf(aOut, "  %zu Processors\n"  , mProcessors.size() );
    fprintf(aOut, "  %zu Threads\n"     , mThreads   .size() );
    fprintf(aOut, "  State       = %s\n", STATE_NAMES[mState]);
    fprintf(aOut, "  System Id   = %u\n", mConnect.mSystemId );

    return OpenNet::STATUS_OK;
}

OpenNet::Kernel * System_Internal::Kernel_Get(unsigned int aIndex)
{
    if (mThreads.size() <= aIndex)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return NULL;
    }

    assert(NULL != mThreads[aIndex]);

    return mThreads[aIndex]->GetKernel();
}

unsigned int System_Internal::Kernel_GetCount() const
{
    return static_cast< unsigned int >( mThreads.size() );
}

OpenNet::Processor * System_Internal::Processor_Get(unsigned int aIndex)
{
    if (mProcessors.size() <= aIndex)
    {
        return NULL;
    }

    return mProcessors[aIndex];
}

unsigned int System_Internal::Processor_GetCount() const
{
    return static_cast<unsigned int>(mProcessors.size());
}

OpenNet::Status System_Internal::Start(unsigned int aFlags)
{
    // printf( __CLASS__ "Start( 0x%08x )\n", aFlags );

    switch (mState)
    {
    case STATE_IDLE :
        break;

    case STATE_RUNNING :
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_ALREADY_STARTED;

    default: assert(false);
    }

    mStartFlags = 0;

    Threads_Release();

    OpenNet::Status lResult = OpenNet::STATUS_NO_ADAPTER_CONNECTED;

    try
    {
        Thread * lThread;

        unsigned int i;

        for (i = 0; i < mAdapters.size(); i++)
        {
            Adapter_Internal * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            if (lAdapter->IsConnected(*this))
            {
                lThread = lAdapter->Thread_Prepare();
                if (NULL != lThread)
                {
                    mThreads.push_back(lThread);
                    lResult = OpenNet::STATUS_OK;
                }
            }
        }

        for (i = 0; i < mProcessors.size(); i++)
        {
            assert(NULL != mProcessors[i]);

            lThread = mProcessors[i]->Thread_Prepare();
            if (NULL != lThread)
            {
                mThreads.push_back(lThread);
                lResult = OpenNet::STATUS_OK;
            }
        }

        mStartFlags = aFlags;

        for (i = 0; i < mThreads.size(); i++)
        {
            mThreads[ i ]->Prepare();
        }

        #ifdef _KMS_WINDOWS_

            if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS))
            {
                mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
            }

        #endif

        for (i = 0; i < mThreads.size(); i++)
        {
            mThreads[ i ]->Start();
        }

        if (OpenNet::STATUS_OK == lResult)
        {
            mState = STATE_RUNNING;
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        lResult = ExceptionToStatus(eE);

        OpenNet::Status lStatus = Stop();
        if ( OpenNet::STATUS_OK != lStatus )
        {
            mDebugLog.Log( __FILE__, __FUNCTION__, __LINE__ );
            mDebugLog.Log( OpenNet::Status_GetName( lStatus ) );
        }
    }

    return lResult;
}

OpenNet::Status System_Internal::Stop()
{
    switch (mState)
    {
    case STATE_IDLE :
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_NOT_STARTED;

    case STATE_RUNNING:
        break;

    default: assert(false);
    }

    assert(0 < mThreads.size());

    mState = STATE_IDLE;

    try
    {
        unsigned int i;

        for (i = 0; i < mThreads.size(); i++)
        {
            mThreads[i]->Stop();
        }

        for (i = 0; i < mThreads.size(); i++)
        {
            if (0 != (mStartFlags & START_FLAG_LOOPBACK))
            {
                mThreads[i]->Stop_Wait(::SendLoopBackPackets, this);
            }
            else
            {
                mThreads[i]->Stop_Wait(NULL, NULL);
            }
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        return ExceptionToStatus(eE);
    }

    #ifdef _KMS_WINDOWS_

        if (!SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS))
        {
            mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        }

    #endif

    return OpenNet::STATUS_OK;
}

// ===== OpenNet::StatisticsProvider ========================================

OpenNet::Status System_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aOutSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_BUFFER_TOO_SMALL;
    }

    memset(aOut, 0, aOutSize_byte);

    if (NULL != aInfo_byte)
    {
        (*aInfo_byte) = 0;
    }

    if (aReset)
    {
        return ResetStatistics();
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status System_Internal::ResetStatistics()
{
    return OpenNet::STATUS_OK;
}

// Internal
/////////////////////////////////////////////////////////////////////////////

// aAdapter [--O;---]
void System_Internal::SendLoopBackPackets(Adapter_Internal * aAdapter)
{
    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        Adapter_Internal * lAdapter = mAdapters[i];
        assert(NULL != lAdapter);

        if (aAdapter != lAdapter)
        {
            lAdapter->SendLoopBackPackets();
        }
    }
}

// Protected
/////////////////////////////////////////////////////////////////////////////

void System_Internal::Cleanup()
{
    switch (mState)
    {
    case STATE_IDLE:
        break;

    case STATE_RUNNING :
        Stop();
        break;

    default: assert(false);
    }

    Threads_Release();

    unsigned int i;
    
    for (i = 0; i < mAdapters.size(); i++)
    {
        // new ==> delete  See FindAdapters
        delete mAdapters[i];
    }

    for (i = 0; i < mProcessors.size(); i++)
    {
        // new ==> delete  See FindProcessors
        delete mProcessors[i];
    }
}

// Private
/////////////////////////////////////////////////////////////////////////////

void System_Internal::SetPacketSize(unsigned int aSize_byte)
{
    assert(PACKET_SIZE_MAX_byte >= aSize_byte);
    assert(PACKET_SIZE_MIN_byte <= aSize_byte);

    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

    mConfig.mPacketSize_byte = aSize_byte;

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        Adapter_Internal * lAdapter = mAdapters[i];
        assert(NULL != lAdapter);

        if (lAdapter->IsConnected(*this))
        {
            lAdapter->SetPacketSize(mConfig.mPacketSize_byte);
        }
    }
}

// aAdapter [---;---]
//
// Threads  Apps
OpenNet::Status System_Internal::Adapter_Validate(OpenNet::Adapter * aAdapter)
{
    if (NULL == aAdapter)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        if (mAdapters[i] == aAdapter)
        {
            return OpenNet::STATUS_OK;
        }
    }

    mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
    return OpenNet::STATUS_INVALID_ADAPTER;
}

OpenNet::Status System_Internal::Config_Apply(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    if (mConfig.mPacketSize_byte != aConfig.mPacketSize_byte)
    {
        try
        {
            SetPacketSize(aConfig.mPacketSize_byte);
        }
        catch (KmsLib::Exception * eE)
        {
            mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
            mDebugLog.Log(eE);

            return ExceptionToStatus(eE);
        }
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status System_Internal::Config_Validate(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    if (PACKET_SIZE_MAX_byte < aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PACKET_SIZE_MIN_byte > aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    return OpenNet::STATUS_OK;
}

void System_Internal::Threads_Release()
{
    // printf( __CLASS__ "Threads_Release()\n" );

    for (unsigned int i = 0; i < mThreads.size(); i++)
    {
        assert(NULL != mThreads[i]);

        mThreads[i]->Delete();
    }

    mThreads.clear();
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aE [---;R--]
//
// Threads  Apps
OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE)
{
    assert(NULL != aE);

    switch (aE->GetCode())
    {
    case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;
    }

    return OpenNet::STATUS_EXCEPTION;
}

void SendLoopBackPackets(void * aThis, Adapter_Internal * aAdapter)
{
    assert(NULL != aThis   );

    System_Internal * lThis = reinterpret_cast<System_Internal *>(aThis);

    lThis->SendLoopBackPackets(aAdapter);
}
