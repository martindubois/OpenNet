
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestC.cpp

#define __CLASS__ "TestC::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNet/Processor.h>
#include <OpenNet/UserBuffer.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== TestLib ============================================================
#include "TestC.h"

// Public
/////////////////////////////////////////////////////////////////////////////

TestC::TestC() : Test("C", TestLib::CODE_REPLY_ON_SEQUENCE_ERROR, MODE_KERNEL)
{
}

// ===== TestLib::Test ======================================================

TestC::~TestC()
{
}

void TestC::Info_Display() const
{
    Connections_Display_1_Card();

    printf(
        "===== Sequence ===============================\n"
        "    Internel   Ethernet   Internal\n"
        "\n"
        "Dropped <-a-\n"
        "             0 <------- 1 <-- Generator\n"
        "        +-b-\n"
        "        |\n"
        "        +-->   ------->   Not received\n"
        "\n"
        "a) No error\n"
        "b) Error\n"
        "\n"
        "Packets    x1    +    x1 = x2\n"
        "\n"
        "===== Bandwidth ==============================\n"
        "                 Send\n"
        "                 1   Read    Write   Total\n"
        "Ethernet         x1                    x1\n"
        "PCIe                  x1      x1       x2\n"
        "Memory - GPU          x1      x1       x2\n"
        "Memory - Main         x1               x1\n"
        "=== Code ============================ Mode ===\n"
        "REPLY_ON_SEQUENCE_ERROR *             KERNEL *\n");
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== TestLib::Test ======================================================

void TestC::AdjustGeneratorConfig(OpenNet::PacketGenerator::Config * aConfig)
{
    assert(NULL != aConfig);

    aConfig->mAllowedIndexRepeat = 16;
    aConfig->mIndexOffset_byte   = 16;
}

unsigned int TestC::Init()
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
            printf( __CLASS__ "Init - Not enough adapter\n");
            lResult = __LINE__;
        }
        else
        {
            lStatus = mAdapters[1]->ResetConfig();
            assert(OpenNet::STATUS_OK == lStatus);

            OpenNet::Status lStatus = GetGenerator(0)->SetAdapter(mAdapters[1]);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    return lResult;
}

unsigned int TestC::Start( unsigned int aFlags )
{
    SetBufferQty(0, GetConfig()->mBufferQty);

    mUserBuffer = mProcessor->AllocateUserBuffer(sizeof(unsigned int) * 64);
    assert(NULL != mUserBuffer);

    OpenNet::Status lStatus = mKernels[0].SetStaticUserArgument(1, mUserBuffer);
    if (OpenNet::STATUS_OK != lStatus)
    {
        printf(__CLASS__ "Start - Kernel::SetStaticUserArgument( ,  ) failed - ");
        OpenNet::Status_Display(lStatus, stdout);
        printf("\n");

        return __LINE__;
    }

    return Test::Start( aFlags );
}

unsigned int TestC::Stop()
{
    assert(NULL != mUserBuffer);

    unsigned int lResult = Test::Stop();
    if (0 == lResult)
    {
        unsigned int lCounters[64];

        OpenNet::Status lStatus = mUserBuffer->Read(0, lCounters, sizeof(lCounters));
        if (OpenNet::STATUS_OK == lStatus)
        {
            for (unsigned int i = 0; i < 64; i++)
            {
                printf(" %u", lCounters[i]);
            }

            printf("\n");
        }
        else
        {
            printf( __CLASS__ "Stop - UserBuffer::Read( , ,  ) failed - " );
            OpenNet::Status_Display(lStatus, stdout);
            printf("\n");

            return __LINE__;
        }

        InitAdapterConstraints();

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1640;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 23000;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   216;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 23000;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   216;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 23000;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   216;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Test::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  7490;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 1 * 1024 * 1024;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1420000;
        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   13800;

        mConstraints[TestLib::Test::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID ].mMax = 51;

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

    mUserBuffer->Delete();

    return lResult;
}
