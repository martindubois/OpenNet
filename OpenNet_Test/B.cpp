
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

    KMS_TEST_COMPARE_RETURN(0, lTD.B(8, 9000, 120.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(82 * 1024 * 1024 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83 * 1024 * 1024 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9566 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9601 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(B_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.B(8, 9000, 120.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(82 * 1024 * 1024 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(83 * 1024 * 1024 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(9560 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(9599 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END
