
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Filter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Filter.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Filter_Base)
{
    OpenNet::Filter lF0;

    KMS_TEST_COMPARE(                                        0, lF0.GetCodeLineCount());
    KMS_TEST_ASSERT (NULL                                    == lF0.GetCodeLines    ());
    KMS_TEST_COMPARE(                                        0, lF0.GetCodeSize     ());
    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_NOT_SET             , lF0.ResetCode       ());
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lF0.SetCode         (NULL, 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_EMPTY_CODE               , lF0.SetCode         ("", 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lF0.SetCode         (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_CANNOT_OPEN_INPUT_FILE   , lF0.SetCode         ("DoesNotExist"));
    KMS_TEST_COMPARE(OpenNet::STATUS_EMPTY_INPUT_FILE         , lF0.SetCode         ("OpenNet_Test/Tests/Empty.txt"));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lF0.SetName         (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lF0.SetName         (""));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lF0.Display         (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lF0.Display         (stdout));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Remove     (NULL));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Remove     (" "));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Replace    (NULL, NULL));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Replace    ("A", NULL));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Replace    ("A", "B"));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Search     (NULL));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Search     (" "));
    KMS_TEST_COMPARE(                             0, strcmp("", lF0.GetName         ()));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lF0.SetCode         ("OpenNet_Test/Filter.cpp"));
    KMS_TEST_COMPARE(                                       88, lF0.GetCodeLineCount());
    KMS_TEST_ASSERT (NULL                                    != lF0.GetCodeLines    ());
    KMS_TEST_COMPARE(                                     5769, lF0.GetCodeSize     ());
    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET         , lF0.SetCode         (" ", 1));
    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET         , lF0.SetCode         ("OpenNet_Test/Filter.cpp"));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Remove     (""));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Replace    ("", ""));
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Replace    ("A\tB", ""));
    KMS_TEST_COMPARE(                                        3, lF0.Edit_Replace    ("Found", "FOUND"));
    KMS_TEST_COMPARE(                                        5, lF0.Edit_Replace    ("FOUND", "FOUN"));
    KMS_TEST_COMPARE(                                     5764, lF0.GetCodeSize     ());
    KMS_TEST_COMPARE(                                        7, lF0.Edit_Replace    ("FOUN", "Found"));
    KMS_TEST_COMPARE(                                     5771, lF0.GetCodeSize     ());
    KMS_TEST_COMPARE(                                        0, lF0.Edit_Search     (""));
    KMS_TEST_COMPARE(                                        7, lF0.Edit_Search     ("Found"));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.ResetCode  ());
    KMS_TEST_COMPARE(                 0, lF0.GetCodeSize());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.SetCode    ("AB", 2));
    KMS_TEST_COMPARE(                 2, lF0.GetCodeSize());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Remove("A"));
    KMS_TEST_COMPARE(                 1, lF0.GetCodeSize());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.ResetCode   ());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.SetCode     ("ABC", 3));
    KMS_TEST_COMPARE(                 3, lF0.GetCodeSize ());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Remove ("B"));
    KMS_TEST_COMPARE(                 2, lF0.GetCodeSize ());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Replace("AC", "ABC"));
    KMS_TEST_COMPARE(                 3, lF0.GetCodeSize ());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Replace("ABC", "ABCD"));
    KMS_TEST_COMPARE(                 4, lF0.GetCodeSize ());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Replace("ABCD", "ABC"));
    KMS_TEST_COMPARE(                 3, lF0.GetCodeSize ());
    KMS_TEST_COMPARE(                 1, lF0.Edit_Remove ("B"));
    KMS_TEST_COMPARE(                 2, lF0.GetCodeSize ());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.ResetCode       ());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lF0.SetCode         ("A\nB\n\rC\rD\r\nE", 11));
    KMS_TEST_COMPARE(                11, lF0.GetCodeSize     ());
    KMS_TEST_COMPARE(                 5, lF0.GetCodeLineCount());
    KMS_TEST_ASSERT (NULL             != lF0.GetCodeLines    ());
}
KMS_TEST_END
