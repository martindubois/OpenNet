
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

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Search(32, 500));
    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(32, 500, lTD.mPacketGenerator_Config.mBandwidth_MiB_s))

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(35.9 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(44.0 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(74852.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(91419.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Function_64B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Search(32, 64));
    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(32, 64, lTD.mPacketGenerator_Config.mBandwidth_MiB_s));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(4.8 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(5.8 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(74821.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(89141.0 >= lTD.Adapter_GetPacketThroughput());
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

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Search(32, 500));
    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(32, 500, lTD.mPacketGenerator_Config.mBandwidth_MiB_s));

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(22.2 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(32.7 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(46375.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(68008.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_64B_SetupC)
{
    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    KMS_TEST_COMPARE_RETURN(0, lTD.A_Search(32, 64));
    KMS_TEST_COMPARE_RETURN(0, lTD.A_Verify(32, 64, lTD.mPacketGenerator_Config.mBandwidth_MiB_s))

    lTD.DisplaySpeed();

    KMS_TEST_ASSERT(2.7 < lTD.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(3.4 > lTD.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(41922.0 <= lTD.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(52348.0 >= lTD.Adapter_GetPacketThroughput());
}
KMS_TEST_END
