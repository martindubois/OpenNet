
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
// Setup-B  A computer with at least 1 adapter and 1 processor. The test
//          program run wihtout administrator privilege. Minimum network
//          trafic is sent to the adapter.
// Setup-C  A computer with at least 2 adapters and 1 processor. The test
//          program run wihtout administrator privilege. Both adapters are
//          connected together
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Display")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-B")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-C")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-C_Release")
KMS_TEST_GROUP_LIST_END

extern int A_Function_9KB_SetupC ();
extern int A_Function_500B_SetupC();
extern int A_Function_64B_SetupC ();
extern int A_Kernel_9KB_SetupC   ();
extern int A_Kernel_500B_SetupC  ();
extern int A_Kernel_64B_SetupC   ();

extern int Adapter_Base   ();
extern int Adapter_Display();
extern int Adapter_SetupA ();

extern int B_Function_9KB_SetupC();
extern int B_Kernel_9KB_SetupC  ();

extern int BlackHole_SetupB();

extern int EthernetAddress_Base();

extern int Kernel_Base   ();
extern int Kernel_Display();

extern int Kernel_Forward_Base   ();
extern int Kernel_Forward_Display();

extern int Mirror_SetupB();

extern int Processor_Base();

extern int SourceCode_Base   ();
extern int SourceCode_Display();

extern int Status_Base();

extern int System_Base   ();
extern int System_Display();
extern int System_SetupA ();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(A_Function_9KB_SetupC , "A - Function - 9 KB - Setup C" , 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(A_Function_500B_SetupC, "A - Function - 500 B - Setup C", 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(A_Function_64B_SetupC , "A - Function - 64 B - Setup C - Release", 5, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(A_Kernel_9KB_SetupC   , "A - Kernel - 9 KB - Setup C"   , 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(A_Kernel_500B_SetupC  , "A - Kernel - 500 B - Setup C"  , 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(A_Kernel_64B_SetupC   , "A - Kernel - 64 B - Setup C - Release"  , 5, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Adapter_Base          , "Adapter - Base"                , 0, 0)
    KMS_TEST_LIST_ENTRY(Adapter_Display       , "Adapter - Display"             , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Adapter_SetupA        , "Adapter - Setup A"             , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(B_Function_9KB_SetupC , "B - Function - 9 KB - Setup C" , 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(B_Kernel_9KB_SetupC   , "B - Kernel - 9 KB - Setup C"   , 4, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(BlackHole_SetupB      , "BlackHole - Setup B"           , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(EthernetAddress_Base  , "EthernetAddress - Base"        , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Base           , "Kernel - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Display        , "Kernel - Display"              , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Kernel_Forward_Base   , "Kernel_Forward - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Forward_Display, "Kernel_Forward - Display"      , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Mirror_SetupB         , "Mirror - Setup B"              , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Processor_Base        , "Processor - Base"              , 0, 0)
    KMS_TEST_LIST_ENTRY(SourceCode_Base       , "SourceCode - Base"             , 0, 0)
    KMS_TEST_LIST_ENTRY(SourceCode_Display    , "SourceCode - Display"          , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Status_Base           , "Status - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(System_Base           , "System - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(System_Display        , "System - Display"              , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(System_SetupA         , "System - Setup A"              , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
KMS_TEST_LIST_END

KMS_TEST_MAIN
