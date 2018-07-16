
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/PacketGenerator_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "PacketGenerator_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const unsigned char PACKET[PACKET_SIZE_MAX_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

static const char * STATE_NAMES[] =
{
    "INIT"    ,
    "RUNNING" ,
    "STOPPING",
};

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static DWORD WINAPI Run(LPVOID aParameter);

// Public
/////////////////////////////////////////////////////////////////////////////

PacketGenerator_Internal::PacketGenerator_Internal() : mAdapter(NULL), mDebugLog("K:\\Dossiers_Actifs\\OpenNet\\DebugLog", "PackeGenerator"), mState(STATE_INIT), mThread(NULL)
{
    memset(&mConfig    , 0, sizeof(mConfig    ));
    memset(&mStatistics, 0, sizeof(mStatistics));

    mConfig.mBandwidth_MiB_s =   50;
    mConfig.mPacketSize_byte = 1024;
}

// ===== OpenNet::PacketGenerator ===========================================

PacketGenerator_Internal::~PacketGenerator_Internal()
{
    OpenNet::Status lStatus;

    switch (mState)
    {
    case STATE_INIT    :
    case STATE_STOPPING:
        break;

    case STATE_RUNNING:
        lStatus = Stop();
        assert(OpenNet::STATUS_OK == lStatus);
        (void)(lStatus);
        break;

    default:
        assert(false);
    }
}

OpenNet::Status PacketGenerator_Internal::GetConfig(Config * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(mConfig));

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::SetAdapter(OpenNet::Adapter * aAdapter)
{
    if (NULL == aAdapter)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mAdapter)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_ADAPTER_ALREADY_SET;
    }

    mAdapter = aAdapter;

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::SetConfig(const Config & aConfig)
{
    if (NULL == (&aConfig))
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    OpenNet::Status lStatus = Config_Validate(aConfig);
    if (OpenNet::STATUS_OK != lStatus)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return lStatus;
    }

    switch (mState)
    {
    case STATE_INIT :
        break;

    case STATE_RUNNING :
    case STATE_STOPPING:
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;

    default: assert(false);
    }

    return Config_Apply(aConfig);
}

OpenNet::Status PacketGenerator_Internal::Display(FILE * aOut)
{
    assert(STATE_QTY > mState);

    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "PacketGenerator :\n");
    fprintf(aOut, "  Adapter   = %s\n", (NULL == mAdapter) ? "Not set" : mAdapter->GetName());
    fprintf(aOut, "  State     = %s\n", STATE_NAMES[mState]);
    fprintf(aOut, "  Thread ID = %u\n", mThreadId);

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Start()
{
    switch (mState)
    {
    case STATE_INIT:
        break;

    case STATE_RUNNING:
    case STATE_STOPPING:
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;

    default: assert(false);
    }

    assert(NULL == mThread);

    mState = STATE_RUNNING;

    mThread = CreateThread(NULL, 0, ::Run, this, 0, &mThreadId);
    if (NULL == mThread)
    {
        mState = STATE_INIT;

        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_THREAD_CLOSE_ERROR;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Stop()
{
    switch (mState)
    {
    case STATE_INIT    :
    case STATE_STOPPING:
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_STOPPED;

    case STATE_RUNNING:
        break;

    default: assert(false);
    }

    assert(NULL != mThread);

    mState = STATE_STOPPING;

    OpenNet::Status lResult = OpenNet::STATUS_OK;

    if (WAIT_OBJECT_0 != WaitForSingleObject(mThread, 1000))
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        lResult = OpenNet::STATUS_THREAD_STOP_TIMEOUT;

        if (!TerminateThread(mThread, __LINE__))
        {
            mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
            lResult = OpenNet::STATUS_THREAD_TERMINATE_ERROR;
        }
    }

    mState = STATE_INIT;

    if (!CloseHandle(mThread))
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        lResult = OpenNet::STATUS_THREAD_CLOSE_ERROR;
    }

    mThread = NULL;

    return lResult;
}

// ===== OpenNet::StatisticsProvider ========================================

OpenNet::Status PacketGenerator_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
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

    unsigned int lCount = aOutSize_byte / sizeof(unsigned int);
    if (OpenNet::PACKET_GENERATOR_STATS_QTY < lCount)
    {
        lCount = OpenNet::PACKET_GENERATOR_STATS_QTY;
    }

    unsigned int lSize_byte = lCount * sizeof(unsigned int);

    memcpy(aOut, &mStatistics, lSize_byte);

    if (NULL != aInfo_byte)
    {
        (*aInfo_byte) = lSize_byte;
    }

    if (aReset)
    {
        return ResetStatistics();
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::ResetStatistics()
{
    mStatistics[OpenNet::PACKET_GENERATOR_STATS_STATISTICS_RESET] ++;

    memset(&mStatistics, 0, sizeof(unsigned int) * OpenNet::PACKET_GENERATOR_STATS_RESET_QTY);

    return OpenNet::STATUS_OK;
}

// Internal
/////////////////////////////////////////////////////////////////////////////

unsigned int PacketGenerator_Internal::Run()
{
    assert(NULL                 != mAdapter                );
    assert(                   0 <  mConfig.mBandwidth_MiB_s);
    assert(                   0 <  mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);

    mStatistics[OpenNet::PACKET_GENERATOR_STATS_RUN_ENTRY] ++;

    double lTemp = mConfig.mBandwidth_MiB_s;

    lTemp *=                   1024.0; // KiB / s
    lTemp *=                   1024.0; // byte / s
    lTemp /= mConfig.mPacketSize_byte; // packet / s
    lTemp /=                   1000.0; // packet / ms
    lTemp *=                     15.6; // packet / cycle

    unsigned int lPacketQty = static_cast<unsigned int>(lTemp + 0.5);
    if (0 >= lPacketQty)
    {
        lPacketQty = 1;
    }

    while (STATE_RUNNING == mState)
    {
        Sleep(10);

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            mStatistics[OpenNet::PACKET_GENERATOR_STATS_SEND_packet] ++;

            OpenNet::Status lStatus = mAdapter->Packet_Send(PACKET, mConfig.mPacketSize_byte);
            if (OpenNet::STATUS_OK != lStatus)
            {
                mStatistics[OpenNet::PACKET_GENERATOR_STATS_SEND_ERROR] ++;
            }
        }

        mStatistics[OpenNet::PACKET_GENERATOR_STATS_SEND_cycle] ++;
    }

    mStatistics[OpenNet::PACKET_GENERATOR_STATS_RUN_EXIT] ++;

    return 0;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aConfig [---;R--]
OpenNet::Status PacketGenerator_Internal::Config_Apply(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Config_Validate(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    if (0 >= aConfig.mBandwidth_MiB_s)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_BANDWIDTH;
    }

    if (0 >= aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (PACKET_SIZE_MAX_byte < aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    return OpenNet::STATUS_OK;
}

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
DWORD WINAPI Run(LPVOID aParameter)
{
    assert(NULL != aParameter);

    PacketGenerator_Internal * lThis = reinterpret_cast<PacketGenerator_Internal *>(aParameter);

    return lThis->Run();
}
