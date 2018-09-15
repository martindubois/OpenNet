
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
#include "../Common/TestLib/TestDual.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(B_Function_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.B(3, 9000, 120.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(81.8 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83.1 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9536.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9671.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(B_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.B(3, 9000, 120.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(81.8 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83.1 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9533.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9671.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END
