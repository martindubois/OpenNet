
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/B.cpp

// Internal   Ethernet   Internal
//
//     +---   <-------   <--- Generator
//     |    0          1
//     +-->   ------->   ---> Dropped

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

KMS_TEST_BEGIN(B_Function_9KB_SetupC)
{
    TestLib::TestDual lTD(8, 8, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s =  120;
    lTD.mPacketGenerator_Config.mPacketSize_byte = 9000;

    KMS_TEST_COMPARE_RETURN(OpenNet::STATUS_OK, lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lTD.mFunctions[0].AddDestination(lTD.mAdapters[0]));

    lTD.Adapter_SetInputFunctions();
    lTD.Start                    ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics();
    lTD.DisplaySpeed        ();

    lTD.Stop();

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12237; // 1248 = 11.4 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 10989;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 150;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 150;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 300;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 300;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1001;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1001;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin = 9600;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13356; // 2437 = 22.3 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 10919;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12238; // 1319 = 12.1 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 10919;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 9600;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = 9600;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 9591; // 24 = 0.25 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 9567;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 9591; // 24 = 0.25 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = 9567;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 9591; // 25 = 0.26 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 9566;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 9591; // 25 = 0.26 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = 9566;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(0));

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12321; // 1964 = 19 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 10357;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 150; // 1 = 0.67 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 149;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 150;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 300;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 300;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 18312; // 1308 = 7.69 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin = 17004;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1002; // 1 = 0 %
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1001;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13371; // 1981 = 17.4 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 11390;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12323; // 1967 = 19 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 10356;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 9600; // 15 = 0.16 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 9585;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 18312; // 1308 = 7.69 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = 17004;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 9601; // 38 = 0.4 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 9563;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 9601; // 17 = 0.18 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = 9564;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 9600; // 23 = 0.24 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 9577;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 9601; // 23 = 0.24 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = 9578;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(1));
}
KMS_TEST_END

KMS_TEST_BEGIN(B_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(8, 8, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s =  120;
    lTD.mPacketGenerator_Config.mPacketSize_byte = 9000;

    KMS_TEST_COMPARE_RETURN(OpenNet::STATUS_OK, lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lTD.mKernels[0].AddDestination(lTD.mAdapters[0]));

    lTD.Adapter_SetInputKernels();
    lTD.Start                  ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics();
    lTD.DisplaySpeed        ();

    lTD.Stop();

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12231;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 10979;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 149;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 149;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 300;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 298;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1002;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin = 9536;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13356;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 12176;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12232;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 10994;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 9536;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = 9536;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 9564;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = 9565;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 9598;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 9561;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = 9560;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(0));

    lTD.Adapter_InitialiseConstraints();

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12351;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 11009;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 149;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 150;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 149;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 300;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 298;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 18312;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin = 17004;

    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1002;
    lTD.mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13370;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 13356;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 12351;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 11011;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 9600;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 9536;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 18312;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin = 17004;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 9560;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 9598;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin = 9560;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 9561;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 83 * 1024 * 1024; // 1 = 1.22 %
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 82 * 1024 * 1024;

    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 9599;
    lTD.mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin = 9560;

    KMS_TEST_COMPARE(0, lTD.Adapter_VerifyStatistics(1));
}
KMS_TEST_END
