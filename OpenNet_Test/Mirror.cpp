
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

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Mirror_SetupA)
{
    OpenNet::Filter_Forward lFF0;

    OpenNet::System * lS0 = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lS0);

    OpenNet::Adapter * lA0 = lS0->Adapter_Get(0);
    KMS_TEST_ASSERT_GOTO(NULL != lA0, Cleanup);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetStats());

    OpenNet::Processor * lP0 = lS0->Processor_Get(0);
    KMS_TEST_ASSERT_GOTO(NULL != lP0, Cleanup);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Adapter_Connect(lA0), Cleanup);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->SetProcessor(lP0), Cleanup);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Display(stdout));

    OpenNet::Adapter::State lState;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->GetState(&lState));
    OpenNet::Adapter::Display(lState, stdout);

    OpenNet::Adapter::Stats lStats;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->GetStats(&lStats));
    OpenNet::Adapter::Display(lStats, stdout);

    lFF0.AddDestination(lA0);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->SetInputFilter(&lFF0));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lFF0.Display(stdout));

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lA0->Buffer_Allocate(3), Cleanup);

    KMS_TEST_COMPARE_GOTO(OpenNet::STATUS_OK, lS0->Start(), Cleanup);

    Sleep(2000);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lS0->Stop());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Buffer_Release(3));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetInputFilter());

Cleanup:

    lS0->Delete();
}
KMS_TEST_END
