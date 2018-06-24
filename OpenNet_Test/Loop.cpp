
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
#include "Utilities.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY (  2)
#define PACKET_QTY (128)

static const uint8_t PACKET[] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
    0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
    0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
};

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Loop_SetupB)
{
    OpenNet::Filter_Forward lFF0;
    OpenNet::Filter_Forward lFF1;

    OpenNet::System * lS0 = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lS0);

    OpenNet::Adapter * lA0 = lS0->Adapter_Get(0);
    OpenNet::Adapter * lA1 = lS0->Adapter_Get(1);
    KMS_TEST_ASSERT_GOTO(NULL != lA0, Cleanup0);
    KMS_TEST_ASSERT_GOTO(NULL != lA1, Cleanup0);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetStats());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA1->ResetStats());

    OpenNet::Processor * lP0 = lS0->Processor_Get(0);
    KMS_TEST_ASSERT_GOTO(NULL != lP0, Cleanup0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Adapter_Connect(lA0), Cleanup0);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Adapter_Connect(lA1), Cleanup0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetProcessor(lP0), Cleanup0);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA1->SetProcessor(lP0), Cleanup0);

    lFF0.AddDestination(lA1);
    lFF1.AddDestination(lA0);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetInputFilter(&lFF0), Cleanup0);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA1->SetInputFilter(&lFF1), Cleanup0);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->Buffer_Allocate(BUFFER_QTY), Cleanup1);
    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA1->Buffer_Allocate(BUFFER_QTY), Cleanup1);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Start(), Cleanup2);

    Sleep(2000);

    for (unsigned int i = 0; i < PACKET_QTY; i++)
    {
        KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Packet_Send(PACKET, sizeof(PACKET)));
        KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA1->Packet_Send(PACKET, sizeof(PACKET)));
    }

    Sleep(2000);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lS0->Stop());

Cleanup2:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Buffer_Release(BUFFER_QTY));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA1->Buffer_Release(BUFFER_QTY));

Cleanup1:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetInputFilter());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA1->ResetInputFilter());

    Sleep(2000);

    OpenNet::Adapter::Stats lStats;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->GetStats(&lStats));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lStats, stdout));

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
    lStatsE.mDriver.mAdapter.mIoCtl_Connect     = 1;
    lStatsE.mDriver.mAdapter.mIoCtl_Packet_Send = PACKET_QTY;
    lStatsE.mDriver.mAdapter.mIoCtl_Start       = 1;
    lStatsE.mDriver.mAdapter.mIoCtl_Stop        = 1;

    lStatsM.mDll.mPacket_Send         = UTL_MASK_ABOVE;
    lStatsM.mDll.mRun_Iteration_Queue = UTL_MASK_ABOVE;
    lStatsM.mDll.mRun_Iteration_Wait  = UTL_MASK_ABOVE;

    lStatsM.mDriver.mAdapter.mBuffers_Process           = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive            = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send               = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl                     = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mIoCtl_State_Get           = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mTx_Packet                 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Last_Result = UTL_MASK_IGNORE;
    lStatsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process        = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2       = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive           = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Send              = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_Packet                = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mTx_Packet                = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware_NoReset.mStats_Reset      = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lStats, lStatsE, lStatsM));

Cleanup0:
    lS0->Delete();
}
KMS_TEST_END
