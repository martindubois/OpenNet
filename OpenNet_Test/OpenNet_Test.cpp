
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/OpenNet_Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_GROUP_LIST_BEGIN
    KMS_TEST_GROUP_LIST_ENTRY("Base")
KMS_TEST_GROUP_LIST_END

extern int System_Base();

KMS_TEST_LIST_BEGIN
    KMS_TEST_LIST_ENTRY(System_Base, "System - Base", 0, 0)
KMS_TEST_LIST_END

KMS_TEST_MAIN
