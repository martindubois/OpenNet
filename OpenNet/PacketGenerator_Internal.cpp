
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
#include "Adapter_Internal.h"
#include "PacketGenerator_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_SIZE_byte ( sizeof(IoCtl_Packet_Send_Ex_In) + PACKET_SIZE_MAX_byte )

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

    mConfig.mBandwidth_MiB_s =   50.0;
    mConfig.mPacketSize_byte = 1024  ;
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

    mAdapter = dynamic_cast<Adapter_Internal *>( aAdapter );
    if (NULL == mAdapter)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_ADAPTER;
    }

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

// Return  This method always return 0 to indicate success.
unsigned int PacketGenerator_Internal::Run()
{
    mStatistics[OpenNet::PACKET_GENERATOR_STATS_RUN_ENTRY] ++;

    unsigned char lBuffer[BUFFER_SIZE_byte];

    IoCtl_Packet_Send_Ex_In * lIn = PreparePacket(lBuffer);
    assert(NULL != lIn);

    double lPeriod = ComputePeriod();
    assert(0.0 < lPeriod);

    LARGE_INTEGER lNow;

    BOOL lRetB = QueryPerformanceCounter(&lNow);
    assert(lRetB);

    while (STATE_RUNNING == mState)
    {
        LARGE_INTEGER lBefore = lNow;

        Sleep(1);

        lRetB = QueryPerformanceCounter(&lNow);
        assert(lRetB);

        lIn->mRepeatCount = ComputeRepeatCount(lBefore, lNow, lPeriod);
        if (0 < lIn->mRepeatCount)
        {
            SendPackets(lIn);
        }
        else
        {
            mStatistics[OpenNet::PACKET_GENERATOR_STATS_NO_PACKET_cycle];
            lNow = lBefore;
        }
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

    if (0.0 >= aConfig.mBandwidth_MiB_s)
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

// Return  This method return the time betweed two packet in clock cycle.
double PacketGenerator_Internal::ComputePeriod() const
{
    LARGE_INTEGER lFrequency;

    BOOL lRetB = QueryPerformanceFrequency(&lFrequency);
    assert(lRetB);

    double lPacketPerSecond = mConfig.mBandwidth_MiB_s;

    lPacketPerSecond *= 1024; // KiB/s
    lPacketPerSecond *= 1024; // B/s
    lPacketPerSecond /= mConfig.mPacketSize_byte; // packet/s

    return lFrequency.QuadPart / lPacketPerSecond;
}

// aBefore [---;R--] The time before the sleep in clock cycle
// aNow    [---;R--] The time after the sleep in clock cycle
// aPeriod           The period in clock cycle
//
// Return  This function return the number of packet to send.
unsigned int PacketGenerator_Internal::ComputeRepeatCount(const LARGE_INTEGER & aBefore, const LARGE_INTEGER & aNow, double aPeriod)
{
    assert(NULL != (&aBefore));
    assert(NULL != (&aNow));
    assert(0.0 <     aPeriod);

    double lDuration = static_cast<double>(aNow.QuadPart - aBefore.QuadPart);
    assert(0.0 < lDuration);

    unsigned int lResult = static_cast<unsigned int>((lDuration / aPeriod) + 0.5);
    if (REPEAT_COUNT_MAX < lResult)
    {
        mStatistics[OpenNet::PACKET_GENERATOR_STATS_TOO_MANY_PACKET_cycle];
        lResult = REPEAT_COUNT_MAX;
    }

    return lResult;
}

// aBuffer [---;-W-] The buffer to initialize
//
// Return  This method return aBuffer converted in IoCtl_Packet_Send_Ex_In
//         pointer.
IoCtl_Packet_Send_Ex_In * PacketGenerator_Internal::PreparePacket(void * aBuffer)
{
    assert(NULL != aBuffer);

    assert(                   0 <  mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);

    memset(aBuffer, 0, BUFFER_SIZE_byte);

    IoCtl_Packet_Send_Ex_In * lResult = reinterpret_cast<IoCtl_Packet_Send_Ex_In *>(aBuffer);

    memcpy(lResult + 1, PACKET, mConfig.mPacketSize_byte);

    return lResult;
}

// aIn [---;R--] The packet including the IoCtl header
void PacketGenerator_Internal::SendPackets(const IoCtl_Packet_Send_Ex_In * aIn)
{
    assert(NULL != aIn              );
    assert(   0 <  aIn->mRepeatCount);

    assert(NULL                 != mAdapter                );
    assert(                   0 <  mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);

    mStatistics[OpenNet::PACKET_GENERATOR_STATS_SEND_cycle] ++;

    try
    {
        mAdapter->Packet_Send_Ex(aIn, sizeof(IoCtl_Packet_Send_Ex_In) + mConfig.mPacketSize_byte);

        mStatistics[OpenNet::PACKET_GENERATOR_STATS_SENT_packet] += aIn->mRepeatCount;
    }
    catch (...)
    {
        mStatistics[OpenNet::PACKET_GENERATOR_STATS_SEND_ERROR_cycle] ++;
    }
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
DWORD WINAPI Run(LPVOID aParameter)
{
    assert(NULL != aParameter);

    PacketGenerator_Internal * lThis = reinterpret_cast<PacketGenerator_Internal *>(aParameter);

    return lThis->Run();
}
