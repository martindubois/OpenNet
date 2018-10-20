
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/B.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNetK/Adapter_Statistics.h"
#include "../Common/TestLib/Tester.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(B_Function_9KB_SetupC)
{
    TestLib::Tester::B_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth ( 120.0);
    lT.SetPacketSize(9000  );

    KMS_TEST_COMPARE_RETURN(0, lT.B(3));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(81.8 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83.1 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9536.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9671.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(B_Kernel_9KB_SetupC)
{
    TestLib::Tester::B_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth ( 120.0);
    lT.SetPacketSize(9000  );

    KMS_TEST_COMPARE_RETURN(0, lT.B(3));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(81.8 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83.1 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9533.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9671.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END
