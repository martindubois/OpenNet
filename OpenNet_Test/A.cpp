
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
#include "../Common/TestLib/Tester.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(A_Function_9KB_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth ( 120.0);
    lT.SetPacketSize(9000  );

    KMS_TEST_COMPARE_RETURN(0, lT.A(2));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(119.0 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(119.1 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(13860.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(13862.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Function_500B_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth (120.0);
    lT.SetPacketSize(500  );

    KMS_TEST_COMPARE_RETURN(0, lT.A(4));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(114.7 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(114.8 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(238665.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(238702.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

// This test verify the maximum packet throughtput when using a loopback
// between the 2 ports of a same network card.
KMS_TEST_BEGIN(A_Function_64B_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth (92.0);
    lT.SetPacketSize(64  );

    KMS_TEST_COMPARE_RETURN(0, lT.Verify('A', 60));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(71.5 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(72.0 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(1103712.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(1109013.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

// This test verify the maximum packet throughput when using a loopback
// between 2 network card.
KMS_TEST_BEGIN(A_Function_64B_SetupD)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_FUNCTION, false);

    lT.SetBandwidth (92.0);
    lT.SetPacketSize(64  );

    KMS_TEST_COMPARE_RETURN(0, lT.Verify('A', 64));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(91.6 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(92.2 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(1413691.3 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(1421280.8 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_9KB_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_KERNEL, false);

    lT.SetBandwidth ( 120.0);
    lT.SetPacketSize(9000  );

    KMS_TEST_COMPARE_RETURN(0, lT.A(2));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(119.0 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(119.1 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(13860.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(13875.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_500B_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_KERNEL, false);

    lT.SetBandwidth (120.0);
    lT.SetPacketSize(500  );

    KMS_TEST_COMPARE_RETURN(0, lT.A(4));

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(114.7 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(114.8 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(238695.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(238723.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END

KMS_TEST_BEGIN(A_Kernel_64B_SetupC)
{
    TestLib::Tester::A_Describe();

    TestLib::Tester lT(TestLib::Tester::MODE_KERNEL, false);

    lT.SetBandwidth (35.0);
    lT.SetPacketSize(64  );

    KMS_TEST_COMPARE_RETURN(0, lT.Verify('A', 24))

    lT.DisplaySpeed();

    KMS_TEST_ASSERT(28.8 < lT.Adapter_GetBandwidth());
    KMS_TEST_ASSERT(37.2 > lT.Adapter_GetBandwidth());

    KMS_TEST_ASSERT(444343.0 <= lT.Adapter_GetPacketThroughput());
    KMS_TEST_ASSERT(572098.0 >= lT.Adapter_GetPacketThroughput());
}
KMS_TEST_END
