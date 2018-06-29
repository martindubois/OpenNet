
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/BlackHole.cpp

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

#define BUFFER_QTY (2)

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(BlackHole_SetupB)
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

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetProcessor   ( lP0      ), Cleanup0);
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

    lStatsM.mDriver.mAdapter.mBuffers_Process     = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_Receive      = UTL_MASK_ABOVE_OR_EQUAL;
    lStatsM.mDriver.mAdapter.mBuffer_Send         = UTL_MASK_ABOVE;
    lStatsM.mDriver.mAdapter.mBuffer_SendPackets  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process  = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mInterrupt_Process2 = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mPacket_Receive     = UTL_MASK_ABOVE;
    lStatsM.mDriver.mHardware.mRx_Packet          = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lStats, lStatsE, lStatsM));

Cleanup0:
    lS0->Delete();
}
KMS_TEST_END
