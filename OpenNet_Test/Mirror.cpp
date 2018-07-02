
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Mirror.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"

// ===== OpenNet_Test =======================================================
#include "SetupA.h"
#include "SetupC.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY       (   2)
#define PACKET_QTY       ( 128)
#define PACKET_SIZE_byte (1500)

static const uint8_t PACKET[PACKET_SIZE_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Constants
/////////////////////////////////////////////////////////////////////////////

#define HARDWARE_BASE (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY)

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Mirror_SetupB)
{
    SetupA lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem ->Adapter_Connect(lSetup.mAdapter  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->SetProcessor   (lSetup.mProcessor));

    lSetup.mFilter.AddDestination(lSetup.mAdapter);

    KMS_TEST_COMPARE(0, lSetup.Statistics_Reset());
    KMS_TEST_COMPARE(0, lSetup.Start           ());

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Stop(0));

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Statistics_Get());

    /*
    OpenNet::Adapter::Stats lStatsE;
    OpenNet::Adapter::Stats lStatsM;

    Utl_ValidateInit(&lStatsE, &lStatsM);

    lStatsE.mDll.mBuffer_Allocated = BUFFER_QTY;
    lStatsE.mDll.mBuffer_Released  = BUFFER_QTY;
    lStatsE.mDll.mRun_Entry        = 1;
    lStatsE.mDll.mRun_Exit         = 1;
    lStatsE.mDll.mRun_Queue        = BUFFER_QTY;
    lStatsE.mDll.mStart            = 1;
    lStatsE.mDll.mStop_Request     = 1;
    lStatsE.mDll.mStop_Wait        = 1;

    lStatsE.mDriver.mAdapter.mBuffer_InitHeader = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Queue      = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Receive    = BUFFER_QTY;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mTx_Packet          = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Send        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mTx_Packet          = UTL_MASK_ABOVE;
    */
}
KMS_TEST_END

KMS_TEST_BEGIN(Mirror_SetupC)
{
    SetupC lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mFilters[0].AddDestination(lSetup.mAdapters[0]));

    KMS_TEST_COMPARE       (0, lSetup.Statistics_Reset());
    KMS_TEST_COMPARE_RETURN(0, lSetup.Start());

    Sleep(1000);

    for (unsigned int i = 0; i < PACKET_QTY; i++)
    {
        KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapters[1]->Packet_Send(PACKET, sizeof(PACKET)));
    }

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Statistics_GetAndDisplay());

    KmsLib::ValueVector::Constraint_UInt32 lConstraints[SetupC::STATISTICS_QTY];

    KmsLib::ValueVector::Constraint_Init(lConstraints, SetupC::STATISTICS_QTY);

    /*
    lStatsE.mDll.mBuffer_Allocated    = BUFFER_QTY;
    lStatsE.mDll.mRun_Entry           = 1;
    lStatsE.mDll.mRun_Iteration_Queue = BUFFER_QTY;
    lStatsE.mDll.mRun_Iteration_Wait  = BUFFER_QTY;
    lStatsE.mDll.mRun_Queue           = BUFFER_QTY;
    lStatsE.mDll.mStart               = 1;

    lStatsE.mDriver.mAdapter.mBuffer_InitHeader  = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Queue       = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Receive     = BUFFER_QTY + (PACKET_QTY / 64);
    lStatsE.mDriver.mAdapter.mBuffer_Send        = PACKET_QTY / 64;
    lStatsE.mDriver.mAdapter.mBuffer_SendPackets = PACKET_QTY / 64 + PACKET_QTY / 64;
    lStatsE.mDriver.mAdapter.mIoCtl              = 5;
    lStatsE.mDriver.mAdapter.mIoCtl_Start        = 1;
    lStatsE.mDriver.mAdapter.mIoCtl_State_Get    = 3;
    lStatsE.mDriver.mAdapter.mTx_Packet          = PACKET_QTY;
    
    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last        = IOCTL_START;
    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last_Result = 0xffffffe0;

    lStatsE.mDriver.mHardware.mInterrupt_Process2 = lStatsE.mDriver.mHardware.mInterrupt_Process;
    lStatsE.mDriver.mHardware.mPacket_Receive     = (BUFFER_QTY * 64) + PACKET_QTY;
    lStatsE.mDriver.mHardware.mPacket_Send        = PACKET_QTY;
    lStatsE.mDriver.mHardware.mRx_Packet          = PACKET_QTY;
    lStatsE.mDriver.mHardware.mTx_Packet          = PACKET_QTY;

    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mTx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Get_Reset = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset     = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mPacket_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mTx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware_NoReset.mStats_Get_Reset = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware_NoReset.mStats_Reset     = UTL_MASK_ABOVE;
    */

    KMS_TEST_COMPARE(0, lSetup.Statistics_Verify(0, lConstraints));

    KmsLib::ValueVector::Constraint_Init(lConstraints, SetupC::STATISTICS_QTY);

    /*
    lStatsE.mDll.mBuffer_Allocated    = BUFFER_QTY;
    lStatsE.mDll.mPacket_Send         = PACKET_QTY;
    lStatsE.mDll.mRun_Entry           = 1;
    lStatsE.mDll.mRun_Iteration_Queue = BUFFER_QTY;
    lStatsE.mDll.mRun_Iteration_Wait  = BUFFER_QTY;
    lStatsE.mDll.mRun_Queue           = BUFFER_QTY;
    lStatsE.mDll.mStart               = 1;

    lStatsE.mDriver.mAdapter.mBuffer_InitHeader  = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Queue       = BUFFER_QTY;
    lStatsE.mDriver.mAdapter.mBuffer_Receive     = BUFFER_QTY + (PACKET_QTY / 64);
    lStatsE.mDriver.mAdapter.mBuffer_Send        = PACKET_QTY / 64;
    lStatsE.mDriver.mAdapter.mBuffer_SendPackets = PACKET_QTY / 64 + PACKET_QTY / 64;
    lStatsE.mDriver.mAdapter.mIoCtl              = PACKET_QTY + 4;
    lStatsE.mDriver.mAdapter.mIoCtl_Packet_Send  = PACKET_QTY;
    lStatsE.mDriver.mAdapter.mIoCtl_Start        = 1;
    lStatsE.mDriver.mAdapter.mIoCtl_State_Get    = 2;

    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last = IOCTL_PACKET_SEND;

    lStatsE.mDriver.mHardware.mInterrupt_Process2 = lStatsE.mDriver.mHardware.mInterrupt_Process;
    lStatsE.mDriver.mHardware.mPacket_Receive     = (BUFFER_QTY * 64) + PACKET_QTY;
    lStatsE.mDriver.mHardware.mPacket_Send        = PACKET_QTY;
    lStatsE.mDriver.mHardware.mRx_Packet          = PACKET_QTY;
    lStatsE.mDriver.mHardware.mTx_Packet          = PACKET_QTY;

    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Get_Reset = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset     = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware_NoReset.mStats_Get_Reset = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware_NoReset.mStats_Reset     = UTL_MASK_ABOVE;
    */
    KMS_TEST_COMPARE(0, lSetup.Statistics_Verify(1, lConstraints));

    KMS_TEST_COMPARE(0, lSetup.Stop(OpenNet::System::STOP_FLAG_LOOPBACK));

    Sleep(1000);

    KMS_TEST_COMPARE(0, lSetup.Statistics_GetAndDisplay());

    KmsLib::ValueVector::Constraint_Init(lConstraints, SetupC::STATISTICS_QTY);

    lConstraints[OpenNet::ADAPTER_STATS_BUFFER_RELEASED].mMin = 2;
    lConstraints[OpenNet::ADAPTER_STATS_BUFFER_RELEASED].mMax = 2;

    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMin        =   0;
    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMax        = 320;
    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMultipleOf =  32;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_EXIT].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_RUN_EXIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_ITERATION_QUEUE].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_ITERATION_WAIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_STOP_REQUEST].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_STOP_REQUEST].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_STOP_WAIT].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_STOP_WAIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 37;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin = 0;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 2;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 2;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 4;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL].mMin =   6;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL].mMax = 327;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax        = 320;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMultipleOf =  64;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATE_GET].mMin = 4;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATE_GET].mMax = 7;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STOP].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STOP].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_TX_packet].mMin        =  64;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_TX_packet].mMax        = 128;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_TX_packet].mMultipleOf =  64;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST].mMin = IOCTL_CONNECT;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST].mMax = IOCTL_STOP   ;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST_RESULT].mMax = 512;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 11;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 28;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 17;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 37;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin        =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax        = 128;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMultipleOf =  64;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin        =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax        = 448;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMultipleOf =  64;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 149;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET].mMax = 1;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin        =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax        = 448;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMultipleOf =  64;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet].mMax = 1;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET_RESET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET_RESET].mMax = 0xffffffff;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_RESET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_RESET].mMax = 0xffffffff;

    KMS_TEST_COMPARE(0, lSetup.Statistics_Verify(0, lConstraints));

    KmsLib::ValueVector::Constraint_Init(lConstraints, SetupC::STATISTICS_QTY);

    lConstraints[OpenNet::ADAPTER_STATS_BUFFER_RELEASED].mMin = 2;
    lConstraints[OpenNet::ADAPTER_STATS_BUFFER_RELEASED].mMax = 2;

    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMin        =   0;
    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMax        = 576;
    lConstraints[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET].mMultipleOf =  64;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_EXIT].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_RUN_EXIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_ITERATION_QUEUE].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_RUN_ITERATION_WAIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_STOP_REQUEST].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_STOP_REQUEST].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_STOP_WAIT].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_STOP_WAIT].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 26;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 43;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 2;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin = 2;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 4;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL].mMin =   6;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL].mMax = 589;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax        = 576;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMultipleOf =  64;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATE_GET].mMin =  4;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATE_GET].mMax = 11;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STOP].mMin = 1;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STOP].mMax = 1;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST].mMin = IOCTL_CONNECT;
    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST].mMax = IOCTL_STOP   ;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_LAST_RESULT].mMax = 512;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

    lConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin = 11;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 34;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMin = 17;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS2].mMax = 43;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin        =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax        = 576;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMultipleOf =  64;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 118;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET].mMax = 1;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin        =  64;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax        = 576;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMultipleOf =  64;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet].mMax = 1;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET_RESET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET_RESET].mMax = 0xffffffff;

    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_RESET].mMin = 1;
    lConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_RESET].mMax = 0xffffffff;

    KMS_TEST_COMPARE(0, lSetup.Statistics_Verify(1, lConstraints));
}
KMS_TEST_END
