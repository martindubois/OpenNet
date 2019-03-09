
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestB.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== TestLib ============================================================
#include "TestB.h"

// Public
/////////////////////////////////////////////////////////////////////////////

TestB::TestB() : Test("B", TestLib::CODE_REPLY, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestB::~TestB()
{
}

void TestB::Info_Display() const
{
    Connections_Display_1_Card();

    printf(
        "===== Sequence ===============================\n"
        "Internel   Ethernet   Internal\n"
        "\n"
        "    +---   <-------   <--- Generator\n"
        "    |    0          1\n"
        "    +-->   ------->\n"
        "\n"
        "Packets x2    +    x1 = x3\n"
        "\n"
        "===== Bandwidth ==============================\n"
        "                 Send\n"
        "                 0   1   Read    Write   Total\n"
        "Ethernet         x1  x1                   x2\n"
        "PCIe                      x2      x1      x3\n"
        "Memory - GPU              x1      x1      x1\n"
        "Memory - Main             x1              x1\n"
        "==============================================\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestB::Init()
{
    assert(NULL == mAdapters[0]);
    assert(NULL == mAdapters[1]);

    SetAdapterCount1(2);
    SetCode         (0, GetConfig()->mCode);

    unsigned int lResult = Test::Init();
    if (0 == lResult)
    {
        assert(NULL != mAdapters[0]);

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = GetSystem()->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf("%s - Not enough adapter\n", __FUNCTION__);
            lResult = __LINE__;
        }
        else
        {
            OpenNet::Status lStatus = GetGenerator(0)->SetAdapter(mAdapters[1]);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    return lResult;
}

unsigned int TestB::Start( unsigned int aFlags )
{
    SetBufferQty(0, GetConfig()->mBufferQty);
    SetBufferQty(0, GetConfig()->mBufferQty);

    return Test::Start( aFlags );
}

unsigned int TestB::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        InitAdapterConstraints();

        unsigned int lBuffer_Max = 42600;
        unsigned int lBuffer_Min =    40;

        unsigned int lPacket_Max = 1430000;
        unsigned int lPacket_Min =    1400;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =        1075;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMax = 1;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMin = 0;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1600;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 11100;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  1000;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 180 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 180 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID ].mMax = 47;

        lResult = VerifyAdapterStats(0);
        if (0 == lResult)
        {
            double lRunningTime_ms = GetAdapterStats(0, TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms);

            mResult.mBandwidth_MiB_s  = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte);
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
