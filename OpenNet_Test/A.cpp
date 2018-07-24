
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/A.cpp

//     Internel   Ethernet   Internal
//
// Dropped <--- 0 <------- 1 <--- Generator

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNetK/Adapter_Statistics.h"
#include "../Common/TestLib/TestDual.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(A_Function_9KB_SetupC)
{
    TestLib::TestDual lTD(8, 2, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s =  120;
    lTD.mPacketGenerator_Config.mPacketSize_byte = 9000;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]));

    lTD.Adapter_SetInputFunction(0);
    lTD.Start                   ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics();
    lTD.DisplaySpeed        ();

    lTD.Stop();

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12674; // 88 = 0.7 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 12586;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1002; // 1 = 0 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1001;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13363; // 778 = 6.18 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 12585;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12674; // 78 = 0.62 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 12596;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 13888;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 13888;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 13889; // 19 = 0.14 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 13870;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 120 * 1024 * 1024; // 1 MiB = 0.84 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 119 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 13888; // 18 = 0.13 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 13870;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(0));
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(8, 2, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s =  120;
    lTD.mPacketGenerator_Config.mPacketSize_byte = 9000;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]));

    lTD.Adapter_SetInputKernel(0);
    lTD.Start                 ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics    ();
    lTD.DisplayAdapterStatistics();
    lTD.DisplaySpeed            ();

    lTD.Stop();

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12663; // 81 = 0.64 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 12582;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 217;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 217;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1001;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1001;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13350; // 5 = 0 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 13345;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12664; // 81 = 0.64 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 12583;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 13888;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 13888;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 13875; // 1 = 0 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 13874;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 120 * 1024 * 1024; // 1 MiB = 0.84 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 119 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 13875; // 1 = 0 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 13874;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(0));
}
KMS_TEST_END
