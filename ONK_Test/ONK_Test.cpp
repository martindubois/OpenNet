
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
// Setup-B  A computer with 2 adapters and a ethernet cable connecting the
//          2 adapters togheter.
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-B")
KMS_TEST_GROUP_LIST_END

extern int Device_SetupA          ();
extern int Device_Hardware_SetupA ();
extern int Device_Tunnel_SetupA   ();
extern int Device_Tunnel_IO_SetupA();
extern int Device_SetupB          ();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(Device_SetupA          , "Device - Setup A"             , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Device_Hardware_SetupA , "Device - Hardware - SetupA"   , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Device_Tunnel_SetupA   , "Device - Tunnel - SetupA"     , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Device_Tunnel_IO_SetupA, "Device - Tunnel - IO - SetupA", 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Device_SetupB          , "Device - Setup B"             , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
KMS_TEST_LIST_END

KMS_TEST_MAIN
