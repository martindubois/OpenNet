
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Mirror.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "SetupC.h"
#include "Utilities.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY       (   2)
#define PACKET_QTY       ( 128)
#define PACKET_SIZE_byte (1500)

static const uint8_t PACKET[PACKET_SIZE_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Mirror_SetupB)
{
    OpenNet::Filter_Forward lFF0;

    OpenNet::System * lS0 = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lS0);

    OpenNet::Adapter * lA0 = lS0->Adapter_Get(0);
    KMS_TEST_ASSERT_GOTO(NULL != lA0, Cleanup0);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetStats());

    OpenNet::Processor * lP0 = lS0->Processor_Get(0);
    KMS_TEST_ASSERT_GOTO(NULL != lP0, Cleanup0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Adapter_Connect(lA0), Cleanup0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetProcessor(lP0), Cleanup0);

    lFF0.AddDestination(lA0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetInputFilter (&lFF0     ), Cleanup0);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->Buffer_Allocate(BUFFER_QTY), Cleanup1);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Start(), Cleanup2);

    Sleep(2000);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lS0->Stop(0));

Cleanup2:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Buffer_Release(BUFFER_QTY));

Cleanup1:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetInputFilter());

    Sleep(2000);

    OpenNet::Adapter::Stats lStats;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->GetStats(&lStats, true));

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

Cleanup0:
    lS0->Delete();
}
KMS_TEST_END

KMS_TEST_BEGIN(Mirror_SetupC)
{
    SetupC lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(0, lSetup.Stats_Reset());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mFilters[0].AddDestination(lSetup.mAdapters[0]));

    KMS_TEST_COMPARE_RETURN(0, lSetup.Start());

    Sleep(1000);

    for (unsigned int i = 0; i < PACKET_QTY; i++)
    {
        KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapters[1]->Packet_Send(PACKET, sizeof(PACKET)));
    }

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Stats_GetAndDisplay());

    OpenNet::Adapter::Stats lStatsE;
    OpenNet::Adapter::Stats lStatsM;

    Utl_ValidateInit(&lStatsE, &lStatsM);

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
    
    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last        = OPEN_NET_IOCTL_START;
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

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mPacket_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mTx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lSetup.mStats[0], lStatsE, lStatsM));

    Utl_ValidateInit(&lStatsE, &lStatsM);

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

    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last = OPEN_NET_IOCTL_PACKET_SEND;

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

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lSetup.mStats[1], lStatsE, lStatsM));

    KMS_TEST_COMPARE(0, lSetup.Stats_Reset());
    KMS_TEST_COMPARE(0, lSetup.Stop       (OpenNet::System::STOP_FLAG_LOOPBACK));

    Sleep(1000);

    KMS_TEST_COMPARE(0, lSetup.Stats_GetAndDisplay());

    Utl_ValidateInit(&lStatsE, &lStatsM);

    lStatsE.mDll.mBuffer_Released = BUFFER_QTY;
    lStatsE.mDll.mRun_Exit        = 1;
    lStatsE.mDll.mStop_Request    = 1;
    lStatsE.mDll.mStop_Wait       = 1;

    lStatsE.mDriver.mAdapter.mBuffer_Send        = 1;
    lStatsE.mDriver.mAdapter.mBuffer_SendPackets = 2;
    lStatsE.mDriver.mAdapter.mIoCtl              = lSetup.mStats[0].mDll.mLoopBackPacket;
    lStatsE.mDriver.mAdapter.mIoCtl_Packet_Send  = lSetup.mStats[0].mDll.mLoopBackPacket;
    lStatsE.mDriver.mAdapter.mIoCtl_State_Get    = 4;
    lStatsE.mDriver.mAdapter.mIoCtl_Stop         = 1;
    lStatsE.mDriver.mAdapter.mTx_Packet          = 64;

    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = 2;

    lStatsE.mDriver.mHardware.mInterrupt_Process2 = lSetup.mStats[0].mDriver.mHardware.mInterrupt_Process;
    lStatsE.mDriver.mHardware.mPacket_Send        = lSetup.mStats[0].mDll.mLoopBackPacket;
    lStatsE.mDriver.mHardware.mTx_Packet          = lSetup.mStats[0].mDriver.mHardware.mPacket_Send;

    lStatsE.mDriver.mHardware_NoReset.mStats_Reset = 2;

    lStatsM.mDll.mLoopBackPacket      = UTL_MASK_IGNORE;
    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_IGNORE;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_IGNORE;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mIoCtl              = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl_State_Get    = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mTx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last_Result = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mPacket_Send        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE_OR_EQUAL;

    KMS_TEST_COMPARE(0, Utl_Validate(lSetup.mStats[0], lStatsE, lStatsM));

    Utl_ValidateInit(&lStatsE, &lStatsM);

    lStatsE.mDll.mBuffer_Released = BUFFER_QTY;
    lStatsE.mDll.mRun_Exit        = 1;
    lStatsE.mDll.mStop_Request    = 1;
    lStatsE.mDll.mStop_Wait       = 1;

    lStatsE.mDriver.mAdapter.mBuffer_Send        = 1;
    lStatsE.mDriver.mAdapter.mBuffer_SendPackets = 2;
    lStatsE.mDriver.mAdapter.mIoCtl              = lSetup.mStats[1].mDll.mLoopBackPacket;
    lStatsE.mDriver.mAdapter.mIoCtl_Packet_Send  = lSetup.mStats[1].mDll.mLoopBackPacket;
    lStatsE.mDriver.mAdapter.mIoCtl_State_Get    = 4;
    lStatsE.mDriver.mAdapter.mIoCtl_Stop         = 1;

    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last        = OPEN_NET_IOCTL_STATE_GET;
    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Last_Result = 0x200;
    lStatsE.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = 2;

    lStatsE.mDriver.mHardware.mInterrupt_Process2 = lSetup.mStats[1].mDriver.mHardware.mInterrupt_Process;
    lStatsE.mDriver.mHardware.mPacket_Send        = lSetup.mStats[1].mDll.mLoopBackPacket;
    lStatsE.mDriver.mHardware.mTx_Packet          = lSetup.mStats[1].mDriver.mHardware.mPacket_Send;

    lStatsE.mDriver.mHardware_NoReset.mStats_Reset = 2;

    lStatsM.mDll.mLoopBackPacket      = UTL_MASK_ABOVE;
    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_IGNORE;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_IGNORE;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mIoCtl              = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl_Packet_Send  = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mIoCtl_State_Get    = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mPacket_Send        = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mTx_Packet          = UTL_MASK_ABOVE_OR_EQUAL;

    lStatsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE_OR_EQUAL;

    KMS_TEST_COMPARE(0, Utl_Validate(lSetup.mStats[1], lStatsE, lStatsM));
}
KMS_TEST_END
