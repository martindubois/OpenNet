
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/PacketGenerator_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

#ifdef _KMS_LINUX_
    // ===== System =========================================================
    #include <sys/signal.h>
#endif

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

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
    "START_REQUESTED",
    "STARTING"       ,
    "STOP_REQUESTED" ,
    "STOPPING",
};

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void ReadPerfCounter(PacketGenerator_Internal::PerfCounter * aOut);

// ===== Entry point ========================================================

#ifdef _KMS_LINUX_
    static void Run(void * aParameter);
#endif

#ifdef _KMS_WINDOWS_
    static DWORD WINAPI Run(LPVOID aParameter);
#endif

// Public
/////////////////////////////////////////////////////////////////////////////

PacketGenerator_Internal::PacketGenerator_Internal() : mAdapter(NULL), mDebugLog("K:\\Dossiers_Actifs\\OpenNet\\DebugLog", "PackeGenerator")
{
    memset(&mConfig    , 0, sizeof(mConfig    ));
    memset(&mStatistics, 0, sizeof(mStatistics));

    SetPriority(PRIORITY_HIGH);

    mConfig.mBandwidth_MiB_s =   50.0;
    mConfig.mPacketSize_byte = 1024  ;
}

// ===== OpenNet::PacketGenerator ===========================================

PacketGenerator_Internal::~PacketGenerator_Internal()
{
    OpenNet::Status lStatus;

    switch (GetState())
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

    switch (GetState())
    {
    case STATE_INIT :
        break;

    case STATE_RUNNING        :
    case STATE_START_REQUESTED:
    case STATE_STARTING       :
    case STATE_STOP_REQUESTED :
    case STATE_STOPPING       :
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;

    default: assert(false);
    }

    return Config_Apply(aConfig);
}

OpenNet::Status PacketGenerator_Internal::Display(FILE * aOut)
{
    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "PacketGenerator :\n");
    fprintf(aOut, "  Adapter   = %s\n", (NULL == mAdapter) ? "Not set" : mAdapter->GetName());
    fprintf(aOut, "  State     = %s\n", STATE_NAMES[GetState()]);
    
    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Start()
{
    OpenNet::Status lResult;

    try
    {
        KmsLib::ThreadBase::Start();

        lResult = OpenNet::STATUS_OK;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);
        lResult = OpenNet::STATUS_EXCEPTION;
    }

    return lResult;
}

OpenNet::Status PacketGenerator_Internal::Stop()
{
    OpenNet::Status lResult;

    try
    {
        bool lRetB = StopAndWait(true, 1000);
        assert(lRetB);

        lResult = OpenNet::STATUS_OK;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);
        lResult = OpenNet::STATUS_EXCEPTION;
    }

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

    #ifdef _KMS_LINUX_
        struct timespec lBefore;
        struct timespec lNow   ;
    #endif
    
    #ifdef _KMS_WINDOWS_
        BOOL lRetB = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        assert(lRetB);

        LARGE_INTEGER lBefore;
        LARGE_INTEGER lNow   ;
    #endif
    
    ReadPerfCounter(&lNow);

    while (IsRunning())
    {
        lBefore = lNow;

        #ifdef _KMS_LINUX_
            usleep(1000);
        #endif
        
        #ifdef _KMS_WINDOWS_
            Sleep(1);
        #endif

        ReadPerfCounter(&lNow);

        lIn->mRepeatCount = ComputeRepeatCount(lBefore, lNow, lPeriod);
        if (0 < lIn->mRepeatCount)
        {
            SendPackets(lIn);
        }
        else
        {
            mStatistics[OpenNet::PACKET_GENERATOR_STATS_NO_PACKET_cycle] ++;
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
    double lPacketPerSecond = mConfig.mBandwidth_MiB_s;

    lPacketPerSecond *= 1024; // KiB/s
    lPacketPerSecond *= 1024; // B/s
    lPacketPerSecond /= mConfig.mPacketSize_byte; // packet/s

    #ifdef _KMS_LINUX_
        return 1000000000 / lPacketPerSecond;
    #endif
    
    #ifdef _KMS_WINDOWS_
        LARGE_INTEGER lFrequency;

        BOOL lRetB = QueryPerformanceFrequency(&lFrequency);
        assert(lRetB);
        
        return lFrequency.QuadPart / lPacketPerSecond;
    #endif
}

// aBefore [---;R--] The time before the sleep in clock cycle
// aNow    [---;R--] The time after the sleep in clock cycle
// aPeriod           The period in clock cycle
//
// Return  This function return the number of packet to send.
unsigned int PacketGenerator_Internal::ComputeRepeatCount(const PerfCounter & aBefore, const PerfCounter & aNow, double aPeriod)
{
    assert(NULL != (&aBefore));
    assert(NULL != (&aNow   ));
    assert(0.0 <     aPeriod );

    double lDuration;
    
    #ifdef _KMS_LINUX_
        lDuration  = aNow.tv_sec - aBefore.tv_sec;
        lDuration *= 1000000000;
        lDuration += aNow.tv_nsec - aBefore.tv_nsec;
    #endif
    
    #ifdef _KMS_WINDOWS_
        lDuration = static_cast<double>(aNow.QuadPart - aBefore.QuadPart);
    #endif
    
    assert(0.0 < lDuration);

    unsigned int lResult = static_cast<unsigned int>((lDuration / aPeriod) + 0.5);
    if (REPEAT_COUNT_MAX < lResult)
    {
        mStatistics[OpenNet::PACKET_GENERATOR_STATS_TOO_MANY_PACKET_cycle] ++;
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

void ReadPerfCounter(PacketGenerator_Internal::PerfCounter * aOut)
{
    assert(NULL != aOut);
 
    #ifdef _KMS_LINUX_
        int lRet = clock_gettime(CLOCK_MONOTONIC, aOut);
        assert(0 == lRet);
    #endif
   
    #ifdef _KMS_WINDOWS_
        BOOL lRetB = QueryPerformanceCounter(aOut);
        assert(lRetB);
    #endif
}

// ===== Entry point ========================================================

#ifdef _LMS_LINUX_

    void Run(void * aParameter)
    {
        assert(NULL != aParameter);
        
        PacketGenerator_Internal * lThis = reinterpret_cast<PacketGenerator_Internal *>(aParameter);
        
        unsigned int lRet = lThis->Run();
        
        pthread_exit(lRet); 
    }
    
#endif

#ifdef _KMS_WINDOWS_

    DWORD WINAPI Run(LPVOID aParameter)
    {
        assert(NULL != aParameter);

        PacketGenerator_Internal * lThis = reinterpret_cast<PacketGenerator_Internal *>(aParameter);

        return lThis->Run();
    }

#endif
