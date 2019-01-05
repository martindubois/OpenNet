
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/BlackHole.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ThreadBase.h>
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Kernel_Forward.h>
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "SetupA.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY (2)

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(BlackHole_SetupB)
{
    SetupA lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->Adapter_Connect(lSetup.mAdapter));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->SetProcessor  ( lSetup.mProcessor));

    KMS_TEST_COMPARE(0, lSetup.Statistics_Reset());

    KMS_TEST_COMPARE(0, lSetup.Start(0));

    KmsLib::ThreadBase::Sleep_s(2);

    KMS_TEST_COMPARE(0, lSetup.Stop());

    KmsLib::ThreadBase::Sleep_s(2);

    KMS_TEST_COMPARE(0, lSetup.Statistics_Get());

    /*
    OpenNet::Adapter::Stats lStatsE;
    OpenNet::Adapter::Stats lStatsM;

    Utl_ValidateInit(&lStatsE, &lStatsM);

    lStatsE.mDll.mBuffer_Allocated = BUFFER_QTY;
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
    */
}
KMS_TEST_END
