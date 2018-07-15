
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SourceCode.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/SourceCode.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(SourceCode_Base)
{
    OpenNet::SourceCode   lSC0;
    OpenNet::SourceCode * lSCNP = NULL;

    KMS_TEST_COMPARE(0, lSC0.GetCodeSize());

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSC0.AppendCode (NULL, 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_EMPTY_CODE               , lSC0.AppendCode (""  , 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , lSC0.AppendCode (*lSCNP));
    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_NOT_SET             , lSC0.ResetCode  ());
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSC0.SetCode    (NULL, 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_EMPTY_CODE               , lSC0.SetCode    (""  , 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSC0.SetCode    (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_CANNOT_OPEN_INPUT_FILE   , lSC0.SetCode    ("DoesNotExist"));
    KMS_TEST_COMPARE(OpenNet::STATUS_EMPTY_INPUT_FILE         , lSC0.SetCode    ("OpenNet_Test/Tests/Empty.txt"));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSC0.SetName    (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lSC0.SetName    (""));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSC0.Display    (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lSC0.Display    (stdout));

    KMS_TEST_COMPARE(0, lSC0.Edit_Remove (NULL));
    KMS_TEST_COMPARE(0, lSC0.Edit_Remove (" "));
    KMS_TEST_COMPARE(0, lSC0.Edit_Replace(NULL, NULL));
    KMS_TEST_COMPARE(0, lSC0.Edit_Replace("A", NULL));
    KMS_TEST_COMPARE(0, lSC0.Edit_Replace("A", "B"));
    KMS_TEST_COMPARE(0, lSC0.Edit_Search (NULL));
    KMS_TEST_COMPARE(0, lSC0.Edit_Search (" "));

    KMS_TEST_COMPARE(0, strcmp("", lSC0.GetName()));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.SetCode("OpenNet_Test/SourceCode.cpp"));

    KMS_TEST_COMPARE(4994, lSC0.GetCodeSize());

    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET, lSC0.SetCode(" ", 1));
    KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET, lSC0.SetCode("OpenNet_Test/Kernel.cpp"));

    KMS_TEST_COMPARE(   0, lSC0.Edit_Remove (""));
    KMS_TEST_COMPARE(   0, lSC0.Edit_Replace("", ""));
    KMS_TEST_COMPARE(   0, lSC0.Edit_Replace("A\tB", ""));
    KMS_TEST_COMPARE(   3, lSC0.Edit_Replace("Found", "FOUND"));
    KMS_TEST_COMPARE(   5, lSC0.Edit_Replace("FOUND", "FOUN"));
    KMS_TEST_COMPARE(4989, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(   7, lSC0.Edit_Replace("FOUN", "Found"));
    KMS_TEST_COMPARE(4996, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(   0, lSC0.Edit_Search (""));
    KMS_TEST_COMPARE(   7, lSC0.Edit_Search ("Found"));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.ResetCode());

    KMS_TEST_COMPARE(0, lSC0.GetCodeSize());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.SetCode   ("A", 1));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.AppendCode("B", 1));

    KMS_TEST_COMPARE(2, lSC0.GetCodeSize());
    KMS_TEST_COMPARE(1, lSC0.Edit_Remove("A"));
    KMS_TEST_COMPARE(1, lSC0.GetCodeSize());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.ResetCode());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.SetCode  ("ABC", 3));

    KMS_TEST_COMPARE(3, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(1, lSC0.Edit_Remove ("B"));
    KMS_TEST_COMPARE(2, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(1, lSC0.Edit_Replace("AC", "ABC"));
    KMS_TEST_COMPARE(3, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(1, lSC0.Edit_Replace("ABC", "ABCD"));
    KMS_TEST_COMPARE(4, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(1, lSC0.Edit_Replace("ABCD", "ABC"));
    KMS_TEST_COMPARE(3, lSC0.GetCodeSize ());
    KMS_TEST_COMPARE(1, lSC0.Edit_Remove ("B"));
    KMS_TEST_COMPARE(2, lSC0.GetCodeSize ());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.ResetCode());
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC0.SetCode  ("A\nB\n\rC\rD\r\nE", 11));

    KMS_TEST_COMPARE(11, lSC0.GetCodeSize());

    OpenNet::SourceCode lSC1;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSC1.AppendCode(lSC0));
}
KMS_TEST_END

KMS_TEST_BEGIN(SourceCode_Display)
{
    OpenNet::SourceCode lSC0;

    lSC0.Display(stdout);

    printf("QUESTION  Is the output OK? (y/n)\n");
    char lLine[1024];
    fgets(lLine, sizeof(lLine), stdin);
    KMS_TEST_COMPARE(0, strncmp("y", lLine, 1));
}
KMS_TEST_END
