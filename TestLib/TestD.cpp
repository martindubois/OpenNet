
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestD.cpp

#define __CLASS__ "TestD"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== TestLib ============================================================
#include "TestD.h"

// Public
/////////////////////////////////////////////////////////////////////////////

TestD::TestD() : Test("D", TestLib::CODE_FORWARD, MODE_FUNCTION)
{
}

// ===== TestLib::Test ======================================================

TestD::~TestD()
{
}

void TestD::Info_Display() const
{
    Connections_Display_1_Card();

    printf(
        "===== Sequence ===============================\n"
        "    Internel   Ethernet   Internal\n"
        "\n"
        "        +--- 0 <------- 2 <-- Generator\n"
        "        |"
        "        +--> 1"
        "\n"
        "Packets     x1    +    x1 = x2\n"
        "\n"
        "===== Bandwidth ==============================\n"
        "                 Send\n"
        "                 1       Read    Write   Total\n"
        "Ethernet         x1                       x1\n"
        "PCIe                      x1      x1      x2\n"
        "Memory - GPU              x1      x1      x2\n"
        "Memory - Main             x1              x1\n"
        "=== Code ============================ Mode ===\n"
        "FORWARD *                           FUNCTION *\n"
        "                                    KERNEL\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

unsigned int TestD::Execute(unsigned int aFlags)
{
    unsigned int lResult = Start(aFlags);
    if (0 == lResult)
    {
        uint8_t         lBuffer[256 * 1024];
        unsigned int    lInfo_byte;
        unsigned int    lReadError = 0;
        OpenNet::Status lStatus;

        if (0 == (aFlags & FLAG_DO_NOT_SLEEP))
        {
            for (unsigned int i = 0; i < 16; i++)
            {
                lStatus = mAdapters[1]->Read(lBuffer, sizeof(lBuffer), &lInfo_byte);
                if (OpenNet::STATUS_OK != lStatus)
                {
                    printf(__CLASS__ "Execute - OpenNet::Adapter::Read( , ,  ) failed - ");
                    OpenNet::Status_Display(lStatus, stdout);
                    printf("\n");
                    lReadError ++;
                    break;
                }
            }
        }

        lStatus = mAdapters[1]->Tx_Disable();
        assert(OpenNet::STATUS_OK == lStatus);

        lResult = Stop();

        lStatus = mAdapters[1]->Tx_Enable();
        assert(OpenNet::STATUS_OK == lStatus);

        if (0 < lReadError)
        {
            lResult = __LINE__;
        }
    }

    return lResult;
}

unsigned int TestD::Init()
{
    assert(NULL == mAdapters[0]);
    assert(NULL == mAdapters[1]);
    assert(NULL == mAdapters[2]);

    SetAdapterCount0 (2);
    SetAdapterCount1 (3);
    SetCode          (0, GetConfig()->mCode);
    SetCode          (1, TestLib::CODE_NONE);
    SetGeneratorCount(1);

    unsigned int lResult = Test::Init();
    if (0 == lResult)
    {
        assert(NULL != mAdapters[0]);

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = GetSystem()->Adapter_Get(OpenNetK::ADAPTER_TYPE_TUNNEL_IO, 0);
        if (NULL == mAdapters[1])
        {
            printf(__CLASS__ "Init - No Tunnel IO\n");
            return __LINE__;
        }

        mAdapters[2] = GetSystem()->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[2])
        {
            printf(__CLASS__ "Init - Not enough adapter\n");
            return __LINE__;
        }

        lStatus = GetGenerator(0)->SetAdapter(mAdapters[2]);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    return lResult;
}

unsigned int TestD::Start( unsigned int aFlags )
{
    SetBufferQty(0, GetConfig()->mBufferQty);
    SetBufferQty(1, 0);

    return Test::Start( aFlags );
}

unsigned int TestD::Stop()
{
    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        InitAdapterConstraints();

        unsigned int lBuffer_Max = 20;
        unsigned int lBuffer_Min =  6;

        unsigned int lPacket_Max = 1100;
        unsigned int lPacket_Min =  300;

        unsigned int lRunningTime_Max = 4600;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10300;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =    50;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = lBuffer_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = lBuffer_Min;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin =             1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 90;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 40;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = lPacket_Min;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = lPacket_Max;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = lPacket_Min;

        lResult = VerifyAdapterStats(0);

            InitAdapterConstraints();

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = lBuffer_Max;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = lBuffer_Min;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = lRunningTime_Max;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin =             1000;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = lPacket_Max;
            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin = lPacket_Min;

            mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = lPacket_Max;
            mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = lPacket_Min;

            lResult += VerifyAdapterStats(1);
            if (0 == lResult)
            {
                double lRunningTime_ms = GetAdapterStats(0, TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms);

                mResult.mBandwidth_MiB_s  = GetAdapterStats(0, TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte  );
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
