
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
// Setup-A  A computer with at least 1 adapter and 1 processor. The test
//          program run wihtout administrator privilege. Minimum network
//          trafic is ent to the adapter.
// Setup-C  A computer with at least 2 adapters and 1 processor. The test
//          program run wihtout administrator privilege. Both adapters are
//          connected together
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-B")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-C")
    KMS_TEST_GROUP_LIST_END

extern int Adapter_Base  ();
extern int Adapter_SetupA();

extern int BlackHole_SetupB();

extern int EthernetAddress_Base();

extern int Filter_Base();

extern int Filter_Forward_Base();

extern int Loop_SetupC();

extern int Mirror_SetupB();
extern int Mirror_SetupC();

extern int Pipe_SetupC();

extern int Processor_Base();

extern int Status_Base();

extern int System_Base  ();
extern int System_SetupA();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(Adapter_Base        , "Adapter - Base"        , 0, 0)
    KMS_TEST_LIST_ENTRY(Adapter_SetupA      , "Adapter - Setup A"     , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(BlackHole_SetupB    , "BlackHole - Setup B"   , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(EthernetAddress_Base, "EthernetAddress - Base", 0, 0)
    KMS_TEST_LIST_ENTRY(Filter_Base         , "Filter - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(Filter_Forward_Base , "Filter_Forward - Base" , 0, 0)
    KMS_TEST_LIST_ENTRY(Loop_SetupC         , "Loop - Setup C"        , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Mirror_SetupB       , "Mirror - Setup B"      , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Mirror_SetupC       , "Mirror - Setup C"      , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Pipe_SetupC         , "Pipe - Setup C"        , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Processor_Base      , "Processor - Base"      , 0, 0)
    KMS_TEST_LIST_ENTRY(Status_Base         , "Status - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(System_Base         , "System - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(System_SetupA       , "System - Setup A"      , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
KMS_TEST_LIST_END

KMS_TEST_MAIN
