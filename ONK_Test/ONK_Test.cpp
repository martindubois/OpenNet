
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Test/ONK_Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

// Setup-A  A computer with at least 1 adapter. The test program run wihtout
//          administrator privilege.
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
KMS_TEST_GROUP_LIST_END

extern int Device_SetupA();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(Device_SetupA, "Device - Setup A", 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
KMS_TEST_LIST_END

KMS_TEST_MAIN
