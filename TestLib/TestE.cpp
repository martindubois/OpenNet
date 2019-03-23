
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestE.cpp

#define __CLASS__ "TestE::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNet/Buffer.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== TestLib ============================================================
#include "TestE.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void ProcessEvent(void * aContext, OpenNetK::Event_Type aType, uint64_t aTimestamp_us, uint32_t aData0, void * aData1);

// Public
/////////////////////////////////////////////////////////////////////////////

TestE::TestE() : Test("E", TestLib::CODE_SIGNAL_EVENT, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestE::~TestE()
{
}

void TestE::Info_Display() const
{
    Connections_Display_2_Cards();

    printf(
        "===== Sequence ===============================\n"
        "    Internel   Ethernet   Internal\n"
        "\n"
        "Dropped <--- 0 <------- 1 <--- Generator\n"
        "\n"
        "Packets     x1    +    x1 = x2\n"
        "\n"
        "===== Bandwidth ==============================\n"
        "                 Send\n"
        "                 1       Read    Write   Total\n"
        "Ethernet         x1                       x1\n"
        "PCIe                      x1+     x1      x2\n"
        "Memory - GPU                +     x1      x1\n"
        "Memory - Main             x1              x1\n"
        "=== Code ============================ Mode ===\n"
        "SIGNAL_EVENT *                      FUNCTION *\n"
        "                                    KERNEL\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestE::Init()
{
    assert(NULL == mAdapters[0]);
    assert(NULL == mAdapters[1]);

    SetAdapterCount1(2);
    SetCode         (0, GetConfig()->mCode);

    unsigned int lResult = Test::Init();
    if (0 == lResult)
    {
        assert(NULL != mAdapters[0]);

        memset(&mCounters, 0, sizeof(mCounters));

        mAdapters[0]->Event_RegisterCallback(ProcessEvent, &mCounters);

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = GetSystem()->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf("%s - Not enough adapters\n", __FUNCTION__);
            return __LINE__;
        }

        lStatus = GetGenerator(0)->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    return lResult;
}

unsigned int TestE::Start( unsigned int aFlags )
{
    SetBufferQty(0, GetConfig()->mBufferQty);
    SetBufferQty(1, GetConfig()->mBufferQty);

    return Test::Start( aFlags );
}

unsigned int TestE::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        printf(
            "Buffer Event = %u\n"
            "Event        = %u\n",
            mCounters.mBufferEvent,
            mCounters.mEvent);

        unsigned int lRunningTime_Max = 1020;

        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =   800;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 10700;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   149;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 10700;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   149;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 19400;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin =             1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 1200;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 683000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9530;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 683000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9530;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 1400 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   15 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1220000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =    9530;

        lResult = VerifyAdapterStats(0);

        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_INTERRUPT_PROCESS_3].mMax = 1200;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_INTERRUPT_PROCESS_3].mMin = 1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_ITERATION].mMax = 200;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_ITERATION].mMin =  90;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMax = 200;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMin = 100;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin =             1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 1200;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  900;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 1250000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   14000;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 1180000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =    9530;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 1400 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =   15 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 1220000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =    9530;

        lResult += VerifyAdapterStats(1);
        if (0 == lResult)
        {
            double lRunningTime_ms = GetAdapterStats(0, TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms);

            mResult.mBandwidth_MiB_s  = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte);;
            mResult.mBandwidth_MiB_s /= 1024.0;
            mResult.mBandwidth_MiB_s /= 1024.0;
            mResult.mBandwidth_MiB_s *= 1000.0;
            mResult.mBandwidth_MiB_s /= lRunningTime_ms;

            mResult.mPacketThroughput_packet_s  = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet);
            mResult.mPacketThroughput_packet_s *= 1000.0;
            mResult.mPacketThroughput_packet_s /= lRunningTime_ms;
        }
    }

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

void ProcessEvent(void * aContext, OpenNetK::Event_Type aType, uint64_t aTimestamp_us, uint32_t aData0, void * aData1)
{
    // printf(__CLASS__ "ProcessEvent( , %u, %u us, %u,  )\n", aType, aTimestamp_us, aData0);

    assert(NULL != aContext);

    TestE::Counters * lCounters = reinterpret_cast<TestE::Counters *>(aContext);

    lCounters->mEvent++;

    switch (aType)
    {
    case OpenNetK::EVENT_TYPE_BUFFER :
        assert(NULL != aData1);

        lCounters->mBufferEvent++;

        OpenNet::Buffer * lBuffer;
        
        lBuffer = reinterpret_cast<OpenNet::Buffer *>(aData1);

        lBuffer->ClearEvent();
        break;

    default: assert(false);
    }
}
