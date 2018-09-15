
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

    KMS_TEST_COMPARE_RETURN(0, lTD.A(2, 9000, 120.0));

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

    KMS_TEST_COMPARE_RETURN(0, lTD.A(4, 500, 120.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(114.7 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(114.8 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(238665.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(238699.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Function_64B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(52, 64, 72.0));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(66.4 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(66.5 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(1025020.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(1025023.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_9KB_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A(2, 9000, 120.0));

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

    KMS_TEST_COMPARE_RETURN(0, lTD.A(4, 500, 120.0));

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

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(24, 64, 35.0))

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(28.8 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(37.2 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(444343.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(572098.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END
