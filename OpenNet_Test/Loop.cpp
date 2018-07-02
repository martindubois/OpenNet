
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Loop.cpp

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

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY       (  2)
#define PACKET_QTY       (128)
#define PACKET_SIZE_byte ( 48)

static const uint8_t PACKET[ PACKET_SIZE_byte ] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Loop_SetupC)
{
    SetupC lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mFilters[0].AddDestination(lSetup.mAdapters[1]));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mFilters[1].AddDestination(lSetup.mAdapters[0]));

    KMS_TEST_COMPARE(0, lSetup.Statistics_Reset());

    KMS_TEST_COMPARE(0, lSetup.Start());
    KMS_TEST_COMPARE(0, lSetup.Packet_Send(PACKET, sizeof(PACKET), PACKET_QTY));

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Stop(0));

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Statistics_GetAndDisplay());
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
    lStatsE.mDriver.mAdapter.mIoCtl_Packet_Send = PACKET_QTY;
    lStatsE.mDriver.mAdapter.mIoCtl_Start       = 1;
    lStatsE.mDriver.mAdapter.mIoCtl_Stop        = 1;

    lStatsM.mDll.mPacket_Send         = UTL_MASK_ABOVE;
    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_ABOVE;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_ABOVE;

    lStatsM.mDriver.mAdapter.mBuffers_Process    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive     = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl              = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl_State_Get    = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mTx_Packet          = UTL_MASK_ABOVE;

    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last            = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last_Result     = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Get_Reset = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset     = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware.mInterrupt_Process   = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive      = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Send         = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_Packet           = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mTx_Packet           = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_NoBuffer_packet  = UTL_MASK_IGNORE;
    lStatsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE;

    lStatsM.mDriver.mHardware_NoReset.mStats_Get_Reset = UTL_MASK_IGNORE;

    KMS_TEST_COMPARE(0, Utl_Validate(lSetup.mStats[0], lStatsE, lStatsM));
    */
}
KMS_TEST_END
