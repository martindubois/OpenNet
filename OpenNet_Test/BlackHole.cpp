
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

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY (2)

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(BlackHole_SetupA)
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

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lS0->Stop());

Cleanup2:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Buffer_Release(BUFFER_QTY));

Cleanup1:
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetInputFilter());

    Sleep(2000);

    OpenNet::Adapter::Stats lStats;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->GetStats(&lStats));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lStats, stdout));

    KMS_TEST_COMPARE(BUFFER_QTY, lStats.mDll.mBuffer_Allocated            );
    KMS_TEST_COMPARE(BUFFER_QTY, lStats.mDll.mBuffer_Released             );
    KMS_TEST_COMPARE(         0, lStats.mDll.mPacket_Send                 );
    KMS_TEST_COMPARE(         1, lStats.mDll.mRun_Entry                   );
    KMS_TEST_COMPARE(         0, lStats.mDll.mRun_Exception               );
    KMS_TEST_COMPARE(         1, lStats.mDll.mRun_Exit                    );
    KMS_TEST_COMPARE(         0, lStats.mDll.mRun_Loop_Exception          );
    KMS_TEST_COMPARE(         0, lStats.mDll.mRun_Loop_UnexpectedException);
    KMS_TEST_COMPARE(BUFFER_QTY, lStats.mDll.mRun_Loop_Wait               );
    KMS_TEST_COMPARE(BUFFER_QTY, lStats.mDll.mRun_Queue                   );
    KMS_TEST_COMPARE(         0, lStats.mDll.mRun_UnexpectedException     );
    KMS_TEST_COMPARE(         1, lStats.mDll.mStart                       );
    KMS_TEST_COMPARE(         1, lStats.mDll.mStop                        );

    KMS_TEST_ASSERT (         0 <  lStats.mDriver.mAdapter.mBuffers_Process   );
    KMS_TEST_COMPARE(BUFFER_QTY  , lStats.mDriver.mAdapter.mBuffer_InitHeader );
    KMS_TEST_ASSERT (         0 <  lStats.mDriver.mAdapter.mBuffer_Process    );
    KMS_TEST_ASSERT (         0 <  lStats.mDriver.mAdapter.mBuffer_Process    );
    KMS_TEST_COMPARE(BUFFER_QTY  , lStats.mDriver.mAdapter.mBuffer_Queue      );
    KMS_TEST_ASSERT (BUFFER_QTY <= lStats.mDriver.mAdapter.mBuffer_Receive    );
    KMS_TEST_ASSERT (         0 <  lStats.mDriver.mAdapter.mBuffer_Send       );
    KMS_TEST_ASSERT (         0 <  lStats.mDriver.mAdapter.mBuffer_SendPackets);
    KMS_TEST_ASSERT (         0 == lStats.mDriver.mAdapter.mTx_Packet         );

    KMS_TEST_ASSERT(0 <  lStats.mDriver.mHardware.mInterrupt_Process );
    KMS_TEST_ASSERT(0 <  lStats.mDriver.mHardware.mInterrupt_Process2);
    KMS_TEST_ASSERT(0 <  lStats.mDriver.mHardware.mPacket_Receive    );
    KMS_TEST_ASSERT(0 == lStats.mDriver.mHardware.mPacket_Send       );
    KMS_TEST_ASSERT(0 <  lStats.mDriver.mHardware.mRx_Packet         );
    KMS_TEST_ASSERT(0 == lStats.mDriver.mHardware.mTx_Packet         );

Cleanup0:
    lS0->Delete();
}
KMS_TEST_END
