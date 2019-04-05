
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/OpenNet_Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

// Setup-A  A computer with at least 1 adapter and 1 processor. The test
//          program run wihtout administrator privilege.
// Setup-C  A computer with 2 adapter connected to each other and 1
//          processor. The test program run wihtout administrator privilege.
KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base"   )
    KMS_TEST_GROUP_LIST_ENTRY("Display")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-A")
    KMS_TEST_GROUP_LIST_ENTRY("Setup-C")
KMS_TEST_GROUP_LIST_END

extern int Adapter_Base   ();
extern int Adapter_Display();
extern int Adapter_SetupA ();

extern int EthernetAddress_Base();

extern int Kernel_Base   ();
extern int Kernel_Display();

extern int Kernel_Forward_Base   ();
extern int Kernel_Forward_Display();

extern int Processor_Base();

extern int SetupTool_Base();

extern int SourceCode_Base   ();
extern int SourceCode_Display();

extern int Status_Base();

extern int System_1Packet();
extern int System_Base   ();
extern int System_Display();
extern int System_SetupA ();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(Adapter_Base          , "Adapter - Base"                , 0, 0)
    KMS_TEST_LIST_ENTRY(Adapter_Display       , "Adapter - Display"             , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Adapter_SetupA        , "Adapter - Setup A"             , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(EthernetAddress_Base  , "EthernetAddress - Base"        , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Base           , "Kernel - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Display        , "Kernel - Display"              , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Kernel_Forward_Base   , "Kernel_Forward - Base"         , 0, 0)
    KMS_TEST_LIST_ENTRY(Kernel_Forward_Display, "Kernel_Forward - Display"      , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Processor_Base        , "Processor - Base"              , 0, 0)
    KMS_TEST_LIST_ENTRY(SetupTool_Base        , "SetupTool - Base"              , 0, 0)
    KMS_TEST_LIST_ENTRY(SourceCode_Base       , "SourceCode - Base"             , 0, 0)
    KMS_TEST_LIST_ENTRY(SourceCode_Display    , "SourceCode - Display"          , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(Status_Base           , "Status - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(System_1Packet        , "System - 1 Packet"             , 3, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(System_Base           , "System - Base"                 , 0, 0)
    KMS_TEST_LIST_ENTRY(System_Display        , "System - Display"              , 1, KMS_TEST_FLAG_INTERACTION_NEEDED)
    KMS_TEST_LIST_ENTRY(System_SetupA         , "System - Setup A"              , 2, KMS_TEST_FLAG_INTERACTION_NEEDED)
KMS_TEST_LIST_END

KMS_TEST_MAIN
