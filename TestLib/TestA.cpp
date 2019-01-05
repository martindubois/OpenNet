
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNetK/Adapter_Statistics.h"

// ===== TestLib ============================================================
#include "TestA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

TestA::TestA() : Test("A", CODE_NOTHING, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestA::~TestA()
{
}

void TestA::Info_Display() const
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
        "==============================================\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestA::Init()
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
            lStatus = GetGenerator(0)->SetAdapter(mAdapters[1]);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    return lResult;
}

unsigned int TestA::Start()
{
    SetBufferQty(0, GetConfig()->mBufferQty);

    return Test::Start();
}

unsigned int TestA::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =   785;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 22100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   217;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 22100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   217;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 22100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   217;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13274;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  5080;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 140 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   13800;

        lResult = VerifyAdapterStats(0);
        if (0 == lResult)
        {
            InitAdapterConstraints();

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 11669;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  5000;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 1540000;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin =   14000;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 11658;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4990;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 1540000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   14000;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 1290000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   13800;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 140 * 1024 * 1024;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =   1 * 1024 * 1024;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 1420000;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   13800;

            lResult = VerifyAdapterStats(1);
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
    }

    return lResult;
}

