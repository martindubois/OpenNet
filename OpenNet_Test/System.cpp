
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "Base.h"
#include "SetupA.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(System_Base)
{
    Base lSetup;

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mSystem->Adapter_Connect(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_ADAPTER          , lSetup.mSystem->Adapter_Connect(reinterpret_cast<OpenNet::Adapter *>(1)));

    KMS_TEST_ASSERT(NULL == lSetup.mSystem->Adapter_Get(0));
    KMS_TEST_COMPARE(0, lSetup.mSystem->Adapter_GetCount());

    KMS_TEST_ASSERT(NULL == lSetup.mSystem->Processor_Get(0));
    KMS_TEST_COMPARE(0, lSetup.mSystem->Processor_GetCount());
}
KMS_TEST_END

// TEST INFO  System
//            Enumerate adapter<br>
//            Enumerate processor
KMS_TEST_BEGIN(System_SetupA)
{
    SetupA lSetup(0);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_ASSERT(0 < lSetup.mSystem->Adapter_GetCount  ());
    KMS_TEST_ASSERT(0 < lSetup.mSystem->Processor_GetCount());
}
KMS_TEST_END
