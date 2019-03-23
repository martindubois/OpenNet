
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== TestLib ============================================================
#include "TestF.h"

// Public
/////////////////////////////////////////////////////////////////////////////

TestF::TestF() : Test("F", TestLib::CODE_FORWARD, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestF::~TestF()
{
}

void TestF::Info_Display() const
{
    Connections_Display_2_Cards();

    printf(
        "===== Sequence ===============================\n"
        "Internel   Ethernet   Internal\n"
        "\n"
        "    +--- 0 <------- 2 <--- Generator\n"
        "    |\n"
        "    +--> 1 -------> 3  ---> Dropped\n"
        "\n"
        "Packets x2    +    x1 = x3\n"
        "\n"
        "===== Bandwidth ==============================\n"
        "                 Send\n"
        "                 1   2   Read    Write   Total\n"
        "Ethernet         x1  x1                   x2\n"
        "PCIe                      x2      x2      x4\n"
        "Memory - GPU              x1      x2      x3\n"
        "Memory - Main             x1              x1\n"
        "==============================================\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestF::Init()
{
    assert(NULL == mAdapters[0]);
    assert(NULL == mAdapters[1]);
    assert(NULL == mAdapters[2]);

    SetAdapterCount0(2);
    SetAdapterCount1(3);
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
            return __LINE__;
        }

        mAdapters[2] = GetSystem()->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_1, MASK_E);
        if (NULL == mAdapters[2])
        {
            printf("%s - Not enough adapter\n", __FUNCTION__);
            return __LINE__;
        }

        lStatus = GetGenerator(0)->SetAdapter(mAdapters[2]);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    return lResult;
}

unsigned int TestF::Start( unsigned int aFlags )
{
    SetBufferQty(0, GetConfig()->mBufferQty);

    return Test::Start( aFlags );
}

unsigned int TestF::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1010;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 12100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =    25;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 12100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =    25;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 12100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =    25;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  1160;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 771000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =  13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 771000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =  13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 771000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =  13800;

        lResult = VerifyAdapterStats(0);
        if (0 == lResult)
        {
            InitAdapterConstraints();

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  8370;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 12100;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   217;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 771000;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin =  13800;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  8650;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 771000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =  13800;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 771000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =  13800;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 771000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =  13800;

            lResult = VerifyAdapterStats(1);
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
    }

    return lResult;
}
