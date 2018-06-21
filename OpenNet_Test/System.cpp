
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/System.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(System_Base)
{
    OpenNet::System * lS0 = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lS0);

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lS0->Adapter_Connect(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_ADAPTER          , lS0->Adapter_Connect(reinterpret_cast<OpenNet::Adapter *>(1)));

    KMS_TEST_ASSERT(NULL == lS0->Adapter_Get(0));
    KMS_TEST_COMPARE(0, lS0->Adapter_GetCount());

    KMS_TEST_ASSERT(NULL == lS0->Processor_Get(0));
    KMS_TEST_COMPARE(0, lS0->Processor_GetCount());

    lS0->Delete();
}
KMS_TEST_END

// TEST INFO  System
//            Enumerate adapter<br>
//            Enumerate processor
KMS_TEST_BEGIN(System_SetupA)
{
    OpenNet::System * lS0 = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lS0);

    KMS_TEST_ASSERT(NULL != lS0->Adapter_Get     (0));
    KMS_TEST_ASSERT(0    <  lS0->Adapter_GetCount());

    KMS_TEST_ASSERT(NULL != lS0->Processor_Get     (0));
    KMS_TEST_ASSERT(0    <  lS0->Processor_GetCount());

    lS0->Delete();
}
KMS_TEST_END
