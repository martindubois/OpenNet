
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2020 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestG.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNetK/Adapter_Statistics.h"

// ===== TestLib ============================================================
#include "TestG.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static unsigned char MULTICAST_ADDRESS[6] = { 0x01, 0x00, 0x5e, 0x00, 0x00, 0x01 };
static unsigned char UNICAST_ADDRESS  [6] = { 0x00, 0x50, 0xb6, 0xe9, 0xdc, 0x04 };

// Public
/////////////////////////////////////////////////////////////////////////////

TestG::TestG() : Test("G", TestLib::CODE_NOTHING, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestG::~TestG()
{
}

void TestG::Info_Display() const
{
    Connections_Display_1_Card();

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
        "                 1   Read    Write   Total\n"
        "Ethernet         x1                   x1\n"
        "PCIe                  x1      x1      x2\n"
        "Memory - GPU                  x1      x1\n"
        "Memory - Main         x1              x1\n"
        "=== Code ============================ Mode ===\n"
        "NOTHING *                           FUNCTION *\n"
        "REPLY_ON_ERROR                      KERNEL\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestG::Init()
{
    assert(NULL == mAdapters[0]);
    assert(NULL == mAdapters[1]);

    SetAdapterCount1(2);
    SetCode(0, GetConfig()->mCode);

    unsigned int lResult = Test::Init();
    if (0 == lResult)
    {
        OpenNet::Adapter * lA = mAdapters[0];
        assert(NULL != lA);

        OpenNet::Adapter::Config lAC;

        OpenNet::Status lStatus = lA->GetConfig(&lAC);
        assert(OpenNet::STATUS_OK == lStatus);

        lAC.mFlags.mMulticastPromiscuousDisable = true;
        lAC.mFlags.mUnicastPromiscuousDisable   = true;

        memcpy(&lAC.mEthernetAddress[0].mAddress, UNICAST_ADDRESS, sizeof(lAC.mEthernetAddress[0].mAddress));

        lStatus = lA->SetConfig(lAC);
        assert(OpenNet::STATUS_OK == lStatus);

        OpenNet::Adapter::Info lInfo;

        lStatus = lA->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = GetSystem()->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf("%s - Not enough adapter\n", __FUNCTION__);
            lResult = __LINE__;
        }
        else
        {
            lStatus = mAdapters[1]->ResetConfig();
            assert(OpenNet::STATUS_OK == lStatus);

            OpenNet::PacketGenerator * lPG = GetGenerator(0);
            assert(NULL != lPG);

            OpenNet::PacketGenerator::Config lPGC;

            lStatus = lPG->GetConfig(&lPGC);
            assert(OpenNet::STATUS_OK == lStatus);

            memcpy(&lPGC.mDestinationEthernet.mAddress, UNICAST_ADDRESS, sizeof(lPGC.mDestinationEthernet.mAddress));

            lStatus = lPG->SetConfig(lPGC);
            assert(OpenNet::STATUS_OK == lStatus);

            lStatus = lPG->SetAdapter(mAdapters[1]);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    return lResult;
}

unsigned int TestG::Start(unsigned int aFlags)
{
    SetBufferQty(0, GetConfig()->mBufferQty);

    return Test::Start(aFlags);
}

unsigned int TestG::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        // DisplayAdapterStats(0);
        // DisplayAdapterStats(1);

        unsigned int lBuffer_Max = 2200;
        unsigned int lBuffer_Min =  230;

        unsigned int lByte_Max = 1400 * 1024 * 1024;
        unsigned int lByte_Min =    1 * 1024 * 1024;

        unsigned int lInterrupt_Max = 11000;
        unsigned int lInterrupt_Min =  1000;

        unsigned int lPacket_Max = 150000;
        unsigned int lPacket_Min =  14000;

        unsigned int lRunningTime_Max_ms = 1100;
        unsigned int lRunningTime_Min_ms = 1000;

        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = lInterrupt_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = lInterrupt_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max_ms;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = lRunningTime_Min_ms;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = lInterrupt_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = lInterrupt_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = lByte_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = lByte_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = lPacket_Min;

        lResult = VerifyAdapterStats(0);

        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = lInterrupt_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = lInterrupt_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_INTERRUPT_PROCESS_3].mMax = lInterrupt_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_INTERRUPT_PROCESS_3].mMin = lInterrupt_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_BREAK].mMax = 8400;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max_ms;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = lRunningTime_Min_ms;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_ITERATION].mMax = 110;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_ITERATION].mMin =  40;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMax = 3100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT].mMin =  140;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = lInterrupt_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = lInterrupt_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = lByte_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = lByte_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = lPacket_Min;

        lResult += VerifyAdapterStats(1);
        if (0 == lResult)
        {
            double lRunningTime_ms = GetAdapterStats(0, TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms);

            mResult.mBandwidth_MiB_s = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte);
            mResult.mBandwidth_MiB_s /= 1024.0;
            mResult.mBandwidth_MiB_s /= 1024.0;
            mResult.mBandwidth_MiB_s *= 1000.0;
            mResult.mBandwidth_MiB_s /= lRunningTime_ms;

            mResult.mPacketThroughput_packet_s = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet);
            mResult.mPacketThroughput_packet_s *= 1000.0;
            mResult.mPacketThroughput_packet_s /= lRunningTime_ms;
        }
    }

    return lResult;
}
