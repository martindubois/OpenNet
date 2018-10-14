
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/A.cpp

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

KMS_TEST_BEGIN(A_Function_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A(2, 9000, 120.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(119.0 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(119.1 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(13860.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(13862.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

// TODO  OpenNet_Test.A
//       Ajouter un test a 1000 B - Function et Kernel

KMS_TEST_BEGIN(A_Function_500B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A(4, 500, 120.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(114.7 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(114.8 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(238665.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(238702.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

// This test verify the maximum packet throughtput when using a loopback
// between the 2 ports of a same network card.
KMS_TEST_BEGIN(A_Function_64B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(60, 64, 92.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(71.5 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(72.0 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(1103712.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(1109013.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

// This test verify the maximum packet throughput when using a loopback
// between 2 network card.
KMS_TEST_BEGIN(A_Function_64B_SetupD)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(64, 64, 92.0, TestLib::TestDual::ADAPTER_SELECT_CARD_DIFF));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(91.6 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(92.2 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(1413691.3 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(1421280.8 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A(2, 9000, 120.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(119.0 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(119.1 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(13860.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(13875.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_500B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A(4, 500, 120.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(114.7 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(114.8 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(238695.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(238723.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_64B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(24, 64, 35.0, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME))

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(28.8 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(37.2 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(444343.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(572098.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END
