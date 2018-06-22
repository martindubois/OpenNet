
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/OpenNet_Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

// Setup-A  A computer with at least 1 adapter and 1 processor. The test
//          program run wihtout administrator privilege.
// Setup-B  A computer with at least 2 adapters and 1 processor. The test
//          program run wihtout administrator privilege.
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-B")
KMS_TEST_GROUP_LIST_END

extern int Adapter_Base  ();
extern int Adapter_SetupA();

extern int BlackHole_SetupA();

extern int EthernetAddress_Base();

extern int Filter_Base();

extern int Filter_Forward_Base();

extern int Loop_SetupB();

extern int Mirror_SetupA();

extern int Processor_Base();

extern int Status_Base();

extern int System_Base  ();
extern int System_SetupA();

extern int VersionInfo_Base();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(Adapter_Base        , "Adapter - Base"        , 0, 0)
    KMS_TEST_LIST_ENTRY(Adapter_SetupA      , "Adapter - SetupA"      , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(BlackHole_SetupA    , "BlackHole - SetupA"    , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(EthernetAddress_Base, "EthernetAddress - Base", 0, 0)
    KMS_TEST_LIST_ENTRY(Filter_Base         , "Filter - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(Filter_Forward_Base , "Filter_Forward - Base" , 0, 0)
    KMS_TEST_LIST_ENTRY(Loop_SetupB         , "Loop - SetupB"         , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Mirror_SetupA       , "Mirror - SetupA"       , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Processor_Base      , "Processor - Base"      , 0, 0)
    KMS_TEST_LIST_ENTRY(Status_Base         , "Status - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(System_Base         , "System - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(System_SetupA       , "System - SetupA"       , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(VersionInfo_Base    , "VersionInfo - Base"    , 0, 0)
KMS_TEST_LIST_END

KMS_TEST_MAIN
